#!/usr/bin/env python3

"""Lossless compressed storage for high-volume ALERT algorithm frames."""

import concurrent.futures
import datetime as dt
import hashlib
import itertools
import json
import math
import pathlib
import queue
import sqlite3
import threading
import zlib
from collections.abc import Sequence


HIGH_VOLUME_STREAMS = (
    "anchor_observations",
    "risk_evidence",
    "clusters",
)
STREAM_ITEM_KEYS = {
    "anchor_observations": "anchors",
    "risk_evidence": "evidences",
    "clusters": "clusters",
}
DATABASE_FILENAME = "algorithm_frames.sqlite3"
STORAGE_SCHEMA_VERSION = 1
CANONICAL_JSON_VERSION = 1
DEFAULT_COMPRESSION_LEVEL = 1
SUPPORTED_STORAGE_BACKENDS = ("jsonl", "sqlite_zlib", "dual")


class AlgorithmFrameStoreError(RuntimeError):
    """Raised when a frame cannot be stored or verified without data loss."""


def normalize_storage_backend(value):
    backend = str(value).strip().lower()
    if backend not in SUPPORTED_STORAGE_BACKENDS:
        raise AlgorithmFrameStoreError(
            "algorithm_storage_backend must be one of {} (got {!r})".format(
                ", ".join(SUPPORTED_STORAGE_BACKENDS), value
            )
        )
    return backend


def canonical_json_bytes(payload):
    """Encode one complete payload deterministically without numeric rounding."""

    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AlgorithmFrameStoreError(
            "Algorithm frame is not finite JSON data: {}".format(exc)
        ) from exc
    return encoded.encode("utf-8")


def _normalize_compression_level(value):
    try:
        compression_level = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AlgorithmFrameStoreError("Invalid zlib compression level") from exc
    if compression_level < 0 or compression_level > 9:
        raise AlgorithmFrameStoreError(
            "zlib compression level must be between 0 and 9"
        )
    return compression_level


def _strict_json_loads(raw_bytes):
    def reject_constant(token):
        raise ValueError("non-finite JSON constant {}".format(token))

    try:
        payload = json.loads(raw_bytes.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AlgorithmFrameStoreError(
            "Stored algorithm frame is not valid finite UTF-8 JSON: {}".format(exc)
        ) from exc
    if not isinstance(payload, dict):
        raise AlgorithmFrameStoreError("Stored algorithm frame is not a JSON object")
    if _contains_nonfinite(payload):
        raise AlgorithmFrameStoreError(
            "Stored algorithm frame contains a non-finite numeric value"
        )
    return payload


def _contains_nonfinite(value):
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _coerce_integer(value, label, required=False):
    if value is None:
        if required:
            raise AlgorithmFrameStoreError("Missing {}".format(label))
        return None
    if isinstance(value, bool):
        raise AlgorithmFrameStoreError("Invalid {}: {!r}".format(label, value))
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AlgorithmFrameStoreError(
            "Invalid {}: {!r}".format(label, value)
        ) from exc
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        numeric = float(result)
    if not math.isfinite(numeric) or numeric != float(result):
        raise AlgorithmFrameStoreError("Invalid {}: {!r}".format(label, value))
    return result


def _time_parts(value, label, required=False):
    if value is None:
        if required:
            raise AlgorithmFrameStoreError("Missing {}".format(label))
        return None, None
    if not isinstance(value, dict):
        raise AlgorithmFrameStoreError("Invalid {}: expected an object".format(label))

    if "secs" in value or "nsecs" in value:
        secs = _coerce_integer(value.get("secs"), label + ".secs", required=True)
        nsecs = _coerce_integer(value.get("nsecs", 0), label + ".nsecs", required=True)
    elif "sec" in value:
        try:
            timestamp = float(value["sec"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise AlgorithmFrameStoreError("Invalid {}.sec".format(label)) from exc
        if not math.isfinite(timestamp):
            raise AlgorithmFrameStoreError("Invalid {}.sec".format(label))
        secs = math.floor(timestamp)
        nsecs = int(round((timestamp - secs) * 1.0e9))
    else:
        if required:
            raise AlgorithmFrameStoreError("Missing {} timestamp fields".format(label))
        return None, None

    if nsecs < 0 or nsecs >= 1000000000:
        raise AlgorithmFrameStoreError(
            "Invalid {}.nsecs: {}".format(label, nsecs)
        )
    return int(secs), int(nsecs)


def _frame_metadata(stream_name, payload):
    if stream_name not in HIGH_VOLUME_STREAMS:
        raise AlgorithmFrameStoreError(
            "Unsupported compressed algorithm stream: {}".format(stream_name)
        )
    if not isinstance(payload, dict):
        raise AlgorithmFrameStoreError("Algorithm frame payload must be a dictionary")

    header = payload.get("header")
    if not isinstance(header, dict):
        raise AlgorithmFrameStoreError("Algorithm frame is missing header")
    stamp_secs, stamp_nsecs = _time_parts(
        header.get("stamp"), "header.stamp", required=True
    )
    header_seq = _coerce_integer(header.get("seq"), "header.seq", required=False)
    recorded_secs, recorded_nsecs = _time_parts(
        payload.get("recorded_at"), "recorded_at", required=False
    )
    reference_epoch = _coerce_integer(
        payload.get("reference_epoch"), "reference_epoch", required=False
    )

    item_key = STREAM_ITEM_KEYS[stream_name]
    items = payload.get(item_key)
    if not isinstance(items, list):
        raise AlgorithmFrameStoreError(
            "Algorithm frame field {!r} must be a list".format(item_key)
        )
    return {
        "header_seq": header_seq,
        "stamp_secs": stamp_secs,
        "stamp_nsecs": stamp_nsecs,
        "recorded_secs": recorded_secs,
        "recorded_nsecs": recorded_nsecs,
        "reference_epoch": reference_epoch,
        "item_count": len(items),
    }


def _prepare_frame(stream_name, payload, compression_level):
    metadata = _frame_metadata(stream_name, payload)
    raw_payload = canonical_json_bytes(payload)
    compressed_payload = zlib.compress(raw_payload, compression_level)
    return {
        "stream_name": stream_name,
        "metadata": metadata,
        "raw_byte_count": len(raw_payload),
        "compressed_byte_count": len(compressed_payload),
        "payload_sha256": hashlib.sha256(raw_payload).hexdigest(),
        "compressed_payload": compressed_payload,
    }


def _create_schema(connection, compression_level):
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS storage_metadata (
            metadata_key TEXT PRIMARY KEY,
            metadata_value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS stream_frames (
            frame_pk INTEGER PRIMARY KEY AUTOINCREMENT,
            stream_name TEXT NOT NULL,
            header_seq INTEGER,
            stamp_secs INTEGER NOT NULL,
            stamp_nsecs INTEGER NOT NULL,
            recorded_secs INTEGER,
            recorded_nsecs INTEGER,
            reference_epoch INTEGER,
            item_count INTEGER NOT NULL CHECK(item_count >= 0),
            codec TEXT NOT NULL,
            raw_byte_count INTEGER NOT NULL CHECK(raw_byte_count >= 0),
            compressed_byte_count INTEGER NOT NULL CHECK(compressed_byte_count >= 0),
            payload_sha256 TEXT NOT NULL CHECK(length(payload_sha256) = 64),
            compressed_payload BLOB NOT NULL,
            UNIQUE(stream_name, stamp_secs, stamp_nsecs)
        );

        CREATE INDEX IF NOT EXISTS idx_stream_frames_source_stamp
            ON stream_frames(stream_name, stamp_secs, stamp_nsecs);
        """
    )
    metadata = {
        "storage_schema_version": str(STORAGE_SCHEMA_VERSION),
        "codec": "zlib",
        "compression_level": str(compression_level),
        "canonical_json_version": str(CANONICAL_JSON_VERSION),
        "recorder_algorithm_schema_version": "2",
        "created_at_iso": dt.datetime.now().isoformat(),
    }
    with connection:
        for key, value in metadata.items():
            connection.execute(
                "INSERT OR IGNORE INTO storage_metadata(metadata_key, metadata_value) "
                "VALUES (?, ?)",
                (key, value),
            )
    _validate_storage_metadata(
        connection,
        expected_compression_level=compression_level,
    )


def _validate_storage_metadata(connection, expected_compression_level=None):
    try:
        metadata = {
            str(key): str(value)
            for key, value in connection.execute(
                "SELECT metadata_key, metadata_value FROM storage_metadata"
            )
        }
    except sqlite3.Error as exc:
        raise AlgorithmFrameStoreError(
            "Could not read algorithm storage metadata: {}".format(exc)
        ) from exc
    expected = {
        "storage_schema_version": str(STORAGE_SCHEMA_VERSION),
        "codec": "zlib",
        "canonical_json_version": str(CANONICAL_JSON_VERSION),
    }
    for key, expected_value in expected.items():
        actual_value = metadata.get(key)
        if actual_value != expected_value:
            raise AlgorithmFrameStoreError(
                "Invalid algorithm storage metadata {}: {!r} != {!r}".format(
                    key, actual_value, expected_value
                )
            )
    try:
        compression_level = int(metadata.get("compression_level", ""))
    except (TypeError, ValueError) as exc:
        raise AlgorithmFrameStoreError(
            "Invalid algorithm storage metadata compression_level"
        ) from exc
    if compression_level < 0 or compression_level > 9:
        raise AlgorithmFrameStoreError(
            "Invalid algorithm storage metadata compression_level"
        )
    if (
        expected_compression_level is not None
        and compression_level != int(expected_compression_level)
    ):
        raise AlgorithmFrameStoreError(
            "Algorithm storage compression level changed from {} to {}".format(
                compression_level, expected_compression_level
            )
        )
    return metadata


def _summary_from_connection(connection):
    _validate_storage_metadata(connection)
    integrity_rows = connection.execute("PRAGMA integrity_check").fetchall()
    integrity_check = "; ".join(str(row[0]) for row in integrity_rows)
    streams = {}
    rows = connection.execute(
        """
        SELECT stream_name,
               COUNT(*),
               COALESCE(SUM(item_count), 0),
               COALESCE(SUM(raw_byte_count), 0),
               COALESCE(SUM(compressed_byte_count), 0)
          FROM stream_frames
         GROUP BY stream_name
         ORDER BY stream_name
        """
    )
    for stream_name, frame_count, item_count, raw_count, compressed_count in rows:
        raw_count = int(raw_count)
        compressed_count = int(compressed_count)
        streams[str(stream_name)] = {
            "frame_count": int(frame_count),
            "item_count": int(item_count),
            "raw_byte_count": raw_count,
            "compressed_byte_count": compressed_count,
            "compression_ratio": (
                float(compressed_count) / float(raw_count) if raw_count else None
            ),
        }
    return {
        "integrity_check": integrity_check,
        "streams": streams,
    }


class CompressedFrameStore:
    """Synchronous SQLite owner used by the recorder's writer thread."""

    def __init__(self, database_path, compression_level=DEFAULT_COMPRESSION_LEVEL):
        self.database_path = pathlib.Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.compression_level = _normalize_compression_level(compression_level)
        self._closed = False
        self._close_summary = None
        try:
            self._connection = sqlite3.connect(str(self.database_path))
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            _create_schema(self._connection, self.compression_level)
        except (sqlite3.Error, OSError) as exc:
            raise AlgorithmFrameStoreError(
                "Could not initialize algorithm frame database {}: {}".format(
                    self.database_path, exc
                )
            ) from exc

    def write_frame(self, stream_name, payload):
        prepared = _prepare_frame(stream_name, payload, self.compression_level)
        return self.write_prepared_frame(prepared)

    def write_prepared_frame(self, prepared):
        if self._closed:
            raise AlgorithmFrameStoreError("Algorithm frame database is closed")
        stream_name = prepared["stream_name"]
        metadata = prepared["metadata"]
        raw_byte_count = prepared["raw_byte_count"]
        compressed_byte_count = prepared["compressed_byte_count"]
        payload_sha256 = prepared["payload_sha256"]
        compressed_payload = prepared["compressed_payload"]
        try:
            with self._connection:
                cursor = self._connection.execute(
                    """
                    INSERT INTO stream_frames(
                        stream_name,
                        header_seq,
                        stamp_secs,
                        stamp_nsecs,
                        recorded_secs,
                        recorded_nsecs,
                        reference_epoch,
                        item_count,
                        codec,
                        raw_byte_count,
                        compressed_byte_count,
                        payload_sha256,
                        compressed_payload
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stream_name,
                        metadata["header_seq"],
                        metadata["stamp_secs"],
                        metadata["stamp_nsecs"],
                        metadata["recorded_secs"],
                        metadata["recorded_nsecs"],
                        metadata["reference_epoch"],
                        metadata["item_count"],
                        "zlib",
                        raw_byte_count,
                        compressed_byte_count,
                        payload_sha256,
                        sqlite3.Binary(compressed_payload),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise AlgorithmFrameStoreError(
                "Duplicate or invalid {} frame at {}.{:09d}: {}".format(
                    stream_name,
                    metadata["stamp_secs"],
                    metadata["stamp_nsecs"],
                    exc,
                )
            ) from exc
        except sqlite3.Error as exc:
            raise AlgorithmFrameStoreError(
                "Could not commit {} frame: {}".format(stream_name, exc)
            ) from exc
        result = dict(metadata)
        result.update(
            {
                "frame_pk": int(cursor.lastrowid),
                "raw_byte_count": raw_byte_count,
                "compressed_byte_count": compressed_byte_count,
                "payload_sha256": payload_sha256,
            }
        )
        return result

    def close(self):
        if self._closed:
            return self._close_summary
        self._closed = True
        try:
            self._connection.commit()
            self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
            summary = _summary_from_connection(self._connection)
            if summary["integrity_check"] != "ok":
                raise AlgorithmFrameStoreError(
                    "SQLite integrity_check failed: {}".format(
                        summary["integrity_check"]
                    )
                )
            self._close_summary = summary
            return summary
        finally:
            self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class AsyncCompressedFrameStore:
    """Bounded, loss-intolerant queue feeding one SQLite owner thread."""

    _STOP = object()

    def __init__(
        self,
        database_path,
        compression_level=DEFAULT_COMPRESSION_LEVEL,
        queue_capacity=32,
        enqueue_timeout_sec=2.0,
    ):
        try:
            queue_capacity = int(queue_capacity)
            enqueue_timeout_sec = float(enqueue_timeout_sec)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AlgorithmFrameStoreError("Invalid asynchronous writer settings") from exc
        if queue_capacity <= 0:
            raise AlgorithmFrameStoreError("queue_capacity must be positive")
        if not math.isfinite(enqueue_timeout_sec) or enqueue_timeout_sec <= 0.0:
            raise AlgorithmFrameStoreError("enqueue_timeout_sec must be positive")

        self.database_path = pathlib.Path(database_path)
        self.compression_level = _normalize_compression_level(compression_level)
        self.enqueue_timeout_sec = enqueue_timeout_sec
        self._queue = queue.Queue(maxsize=queue_capacity)
        self._state_lock = threading.Lock()
        self._ready = threading.Event()
        self._accepting = True
        self._close_started = False
        self._closed = False
        self._error = None
        self._summary = None
        self._thread = threading.Thread(
            target=self._run,
            name="algorithm-frame-store-writer",
            daemon=False,
        )
        self._thread.start()
        if not self._ready.wait(timeout=30.0):
            self._set_error(
                AlgorithmFrameStoreError(
                    "Timed out initializing the algorithm frame writer"
                )
            )
        self.raise_if_failed()
        self._encoder_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(HIGH_VOLUME_STREAMS),
            thread_name_prefix="algorithm-frame-encoder",
        )

    def _set_error(self, error):
        if not isinstance(error, AlgorithmFrameStoreError):
            wrapped = AlgorithmFrameStoreError(
                "Asynchronous algorithm frame write failed: {}".format(error)
            )
            wrapped.__cause__ = error
            error = wrapped
        with self._state_lock:
            if self._error is None:
                self._error = error
            self._accepting = False

    def _run(self):
        store = None
        try:
            store = CompressedFrameStore(
                self.database_path,
                compression_level=self.compression_level,
            )
        except BaseException as exc:  # propagated to the recorder thread
            self._set_error(exc)
            self._ready.set()
            return

        self._ready.set()
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is self._STOP:
                        break
                    prepared_future, on_commit = item
                    with self._state_lock:
                        prior_error = self._error
                    if prior_error is not None:
                        prepared_future.cancel()
                        continue
                    prepared = prepared_future.result()
                    result = store.write_prepared_frame(prepared)
                    if on_commit is not None:
                        on_commit(result)
                except BaseException as exc:  # retained and raised by close()
                    self._set_error(exc)
                finally:
                    self._queue.task_done()
        finally:
            try:
                self._summary = store.close()
            except BaseException as exc:
                self._set_error(exc)
            with self._state_lock:
                self._closed = True

    def raise_if_failed(self):
        with self._state_lock:
            error = self._error
        if error is not None:
            raise AlgorithmFrameStoreError(str(error)) from error

    def enqueue(self, stream_name, payload, on_commit=None):
        with self._state_lock:
            accepting = self._accepting
            error = self._error
        if error is not None:
            raise AlgorithmFrameStoreError(str(error)) from error
        if not accepting:
            raise AlgorithmFrameStoreError("Algorithm frame writer is closed")
        prepared_future = self._encoder_pool.submit(
            _prepare_frame,
            stream_name,
            payload,
            self.compression_level,
        )
        try:
            self._queue.put(
                (prepared_future, on_commit),
                block=True,
                timeout=self.enqueue_timeout_sec,
            )
        except queue.Full as exc:
            prepared_future.cancel()
            error = AlgorithmFrameStoreError(
                "Algorithm frame writer queue remained full for {:.3f} s".format(
                    self.enqueue_timeout_sec
                )
            )
            self._set_error(error)
            raise error from exc
        self.raise_if_failed()

    def close(self):
        with self._state_lock:
            first_close = not self._close_started
            self._close_started = True
            self._accepting = False

        if first_close:
            while self._thread.is_alive():
                try:
                    self._queue.put(self._STOP, block=True, timeout=0.1)
                    break
                except queue.Full:
                    continue
            self._thread.join()
            self._encoder_pool.shutdown(wait=True)
        elif self._thread.is_alive():
            self._thread.join()

        self.raise_if_failed()
        return self._summary


def _open_read_only(database_path):
    path = pathlib.Path(database_path).resolve()
    if not path.is_file():
        raise AlgorithmFrameStoreError(
            "Algorithm frame database does not exist: {}".format(path)
        )
    try:
        return sqlite3.connect("file:{}?mode=ro".format(path), uri=True)
    except sqlite3.Error as exc:
        raise AlgorithmFrameStoreError(
            "Could not open algorithm frame database {}: {}".format(path, exc)
        ) from exc


def _decode_verified_row(stream_name, row):
    (
        header_seq,
        stamp_secs,
        stamp_nsecs,
        recorded_secs,
        recorded_nsecs,
        reference_epoch,
        item_count,
        codec,
        raw_byte_count,
        compressed_byte_count,
        payload_sha256,
        compressed_payload,
    ) = row
    if codec != "zlib":
        raise AlgorithmFrameStoreError(
            "Unsupported codec {!r} in stream {}".format(codec, stream_name)
        )
    compressed_payload = bytes(compressed_payload)
    if len(compressed_payload) != int(compressed_byte_count):
        raise AlgorithmFrameStoreError(
            "Compressed byte count mismatch in stream {}".format(stream_name)
        )
    try:
        raw_payload = zlib.decompress(compressed_payload)
    except zlib.error as exc:
        raise AlgorithmFrameStoreError(
            "Could not decompress stream {} frame: {}".format(stream_name, exc)
        ) from exc
    if len(raw_payload) != int(raw_byte_count):
        raise AlgorithmFrameStoreError(
            "Raw byte count mismatch in stream {}".format(stream_name)
        )
    actual_sha256 = hashlib.sha256(raw_payload).hexdigest()
    if actual_sha256 != str(payload_sha256):
        raise AlgorithmFrameStoreError(
            "SHA-256 mismatch in stream {}".format(stream_name)
        )
    payload = _strict_json_loads(raw_payload)
    metadata = _frame_metadata(stream_name, payload)
    if metadata["item_count"] != int(item_count):
        raise AlgorithmFrameStoreError(
            "Item count mismatch in stream {}".format(stream_name)
        )
    stored_metadata = {
        "header_seq": header_seq,
        "stamp_secs": stamp_secs,
        "stamp_nsecs": stamp_nsecs,
        "recorded_secs": recorded_secs,
        "recorded_nsecs": recorded_nsecs,
        "reference_epoch": reference_epoch,
    }
    for key, stored_value in stored_metadata.items():
        normalized_stored = None if stored_value is None else int(stored_value)
        if metadata[key] != normalized_stored:
            raise AlgorithmFrameStoreError(
                "Frame metadata mismatch in stream {} for {}: {} != {}".format(
                    stream_name,
                    key,
                    normalized_stored,
                    metadata[key],
                )
            )
    return payload


def iter_sqlite_stream(database_path, stream_name):
    if stream_name not in HIGH_VOLUME_STREAMS:
        raise AlgorithmFrameStoreError(
            "Unsupported compressed algorithm stream: {}".format(stream_name)
        )
    connection = _open_read_only(database_path)
    try:
        try:
            _validate_storage_metadata(connection)
            cursor = connection.execute(
                """
                SELECT header_seq,
                       stamp_secs,
                       stamp_nsecs,
                       recorded_secs,
                       recorded_nsecs,
                       reference_epoch,
                       item_count,
                       codec,
                       raw_byte_count,
                       compressed_byte_count,
                       payload_sha256,
                       compressed_payload
                  FROM stream_frames
                 WHERE stream_name = ?
                 ORDER BY stamp_secs, stamp_nsecs, frame_pk
                """,
                (stream_name,),
            )
            for row in cursor:
                yield _decode_verified_row(stream_name, row)
        except sqlite3.Error as exc:
            raise AlgorithmFrameStoreError(
                "Could not read stream {}: {}".format(stream_name, exc)
            ) from exc
    finally:
        connection.close()


def inspect_sqlite_store(database_path):
    connection = _open_read_only(database_path)
    try:
        try:
            return _summary_from_connection(connection)
        except sqlite3.Error as exc:
            raise AlgorithmFrameStoreError(
                "Could not inspect algorithm frame database: {}".format(exc)
            ) from exc
    finally:
        connection.close()


def iter_jsonl_stream(path, required=False):
    """Yield strict JSON objects without loading the complete text file."""

    path = pathlib.Path(path)
    if not path.is_file():
        if required:
            raise AlgorithmFrameStoreError(
                "Required algorithm JSONL stream does not exist: {}".format(path)
            )
        return
    try:
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    yield _strict_json_loads(stripped)
                except AlgorithmFrameStoreError as exc:
                    raise AlgorithmFrameStoreError(
                        "Invalid JSONL frame at {}:{}: {}".format(
                            path, line_number, exc
                        )
                    ) from exc
    except OSError as exc:
        raise AlgorithmFrameStoreError(
            "Could not read algorithm JSONL stream {}: {}".format(path, exc)
        ) from exc


def _load_storage_contract(run_dir):
    run_dir = pathlib.Path(run_dir)
    run_info_path = run_dir / "meta" / "run_info.json"
    if not run_info_path.is_file():
        return {
            "schema_version": 0,
            "backend": "jsonl",
            "database_path": run_dir / "algorithm" / DATABASE_FILENAME,
            "compressed_streams": set(HIGH_VOLUME_STREAMS),
        }
    try:
        run_info = _strict_json_loads(run_info_path.read_bytes())
    except OSError as exc:
        raise AlgorithmFrameStoreError(
            "Could not read run metadata {}: {}".format(run_info_path, exc)
        ) from exc
    algorithm_recording = run_info.get("algorithm_recording", {})
    if not isinstance(algorithm_recording, dict):
        raise AlgorithmFrameStoreError("run_info.algorithm_recording is not an object")
    schema_version = _coerce_integer(
        algorithm_recording.get("schema_version", 0),
        "algorithm_recording.schema_version",
        required=True,
    )
    raw_backend = algorithm_recording.get("storage_backend")
    if raw_backend is None:
        if schema_version >= 3:
            raise AlgorithmFrameStoreError(
                "Schema-v3 run does not declare algorithm storage_backend"
            )
        backend = "jsonl"
    else:
        backend = normalize_storage_backend(raw_backend)

    raw_database_file = algorithm_recording.get(
        "database_file", "algorithm/{}".format(DATABASE_FILENAME)
    )
    relative_database_path = pathlib.Path(str(raw_database_file))
    if relative_database_path.is_absolute():
        raise AlgorithmFrameStoreError("database_file must be relative to the run")
    database_path = (run_dir / relative_database_path).resolve()
    try:
        database_path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise AlgorithmFrameStoreError(
            "database_file escapes the run directory"
        ) from exc

    raw_streams = algorithm_recording.get(
        "compressed_streams", list(HIGH_VOLUME_STREAMS)
    )
    if not isinstance(raw_streams, list):
        raise AlgorithmFrameStoreError("compressed_streams must be a list")
    compressed_streams = {str(item) for item in raw_streams}
    unknown_streams = compressed_streams - set(HIGH_VOLUME_STREAMS)
    if unknown_streams:
        raise AlgorithmFrameStoreError(
            "Unsupported compressed_streams: {}".format(
                ", ".join(sorted(unknown_streams))
            )
        )
    return {
        "schema_version": schema_version,
        "backend": backend,
        "database_path": database_path,
        "compressed_streams": compressed_streams,
    }


def required_algorithm_storage_paths(run_dir, stream_name):
    """Return run-relative physical files required for one logical stream."""

    run_dir = pathlib.Path(run_dir).resolve()
    stream_name = str(stream_name)
    if stream_name not in HIGH_VOLUME_STREAMS:
        return (pathlib.Path("algorithm") / "{}.jsonl".format(stream_name),)

    contract = _load_storage_contract(run_dir)
    backend = contract["backend"]
    if backend in ("sqlite_zlib", "dual") and stream_name not in contract[
        "compressed_streams"
    ]:
        raise AlgorithmFrameStoreError(
            "Declared backend does not include stream {}".format(stream_name)
        )

    paths = []
    if backend in ("jsonl", "dual"):
        paths.append(pathlib.Path("algorithm") / "{}.jsonl".format(stream_name))
    if backend in ("sqlite_zlib", "dual"):
        try:
            paths.append(contract["database_path"].relative_to(run_dir))
        except ValueError as exc:
            raise AlgorithmFrameStoreError(
                "database_file escapes the run directory"
            ) from exc
    return tuple(paths)


def _resolve_algorithm_stream_source(
    run_dir,
    stream_name,
    representation=None,
):
    run_dir = pathlib.Path(run_dir)
    stream_name = str(stream_name)
    if not stream_name or stream_name in (".", "..") or "/" in stream_name:
        raise AlgorithmFrameStoreError(
            "Invalid logical algorithm stream name: {!r}".format(stream_name)
        )
    jsonl_path = run_dir / "algorithm" / "{}.jsonl".format(stream_name)
    if stream_name not in HIGH_VOLUME_STREAMS:
        if representation is not None and representation != "jsonl":
            raise AlgorithmFrameStoreError(
                "Small algorithm streams are stored only as JSONL"
            )
        return "jsonl", jsonl_path

    contract = _load_storage_contract(run_dir)
    backend = contract["backend"]
    database_path = contract["database_path"]
    if backend in ("sqlite_zlib", "dual") and stream_name not in contract[
        "compressed_streams"
    ]:
        raise AlgorithmFrameStoreError(
            "Declared backend does not include stream {}".format(stream_name)
        )

    has_jsonl = jsonl_path.is_file()
    has_database = database_path.is_file()
    if backend != "dual" and has_jsonl and has_database:
        raise AlgorithmFrameStoreError(
            "Ambiguous representations for stream {}; declare dual mode".format(
                stream_name
            )
        )

    if representation is not None:
        representation = normalize_storage_backend(representation)
        if representation == "dual":
            raise AlgorithmFrameStoreError(
                "representation must be jsonl or sqlite_zlib"
            )
        if backend != "dual" and representation != backend:
            raise AlgorithmFrameStoreError(
                "Requested representation {} conflicts with declared backend {}".format(
                    representation, backend
                )
            )
        selected = representation
    else:
        selected = "sqlite_zlib" if backend in ("sqlite_zlib", "dual") else "jsonl"

    return selected, jsonl_path if selected == "jsonl" else database_path


def _count_jsonl_stream(path, required=False):
    path = pathlib.Path(path)
    if not path.is_file():
        if required:
            raise AlgorithmFrameStoreError(
                "Required algorithm JSONL stream does not exist: {}".format(path)
            )
        return 0
    try:
        with path.open("rb") as handle:
            return sum(1 for raw_line in handle if raw_line.strip())
    except OSError as exc:
        raise AlgorithmFrameStoreError(
            "Could not read algorithm JSONL stream {}: {}".format(path, exc)
        ) from exc


def _count_sqlite_stream(database_path, stream_name):
    connection = _open_read_only(database_path)
    try:
        try:
            _validate_storage_metadata(connection)
            row = connection.execute(
                "SELECT COUNT(*) FROM stream_frames WHERE stream_name = ?",
                (stream_name,),
            ).fetchone()
            return int(row[0])
        except sqlite3.Error as exc:
            raise AlgorithmFrameStoreError(
                "Could not count stream {}: {}".format(stream_name, exc)
            ) from exc
    finally:
        connection.close()


def count_algorithm_stream(
    run_dir,
    stream_name,
    required=False,
    representation=None,
):
    """Count logical frames without decoding complete payloads."""

    selected, source_path = _resolve_algorithm_stream_source(
        run_dir,
        stream_name,
        representation=representation,
    )
    if selected == "jsonl":
        return _count_jsonl_stream(source_path, required=required)
    if not source_path.is_file():
        if required:
            raise AlgorithmFrameStoreError(
                "Required algorithm frame database does not exist: {}".format(
                    source_path
                )
            )
        return 0
    return _count_sqlite_stream(source_path, str(stream_name))


def iter_algorithm_stream(
    run_dir,
    stream_name,
    required=False,
    representation=None,
):
    """Yield a logical algorithm stream from its declared representation."""

    selected, source_path = _resolve_algorithm_stream_source(
        run_dir,
        stream_name,
        representation=representation,
    )

    if selected == "jsonl":
        yield from iter_jsonl_stream(source_path, required=required)
        return
    if not source_path.is_file():
        if required:
            raise AlgorithmFrameStoreError(
                "Required algorithm frame database does not exist: {}".format(
                    source_path
                )
            )
        return
    yield from iter_sqlite_stream(source_path, str(stream_name))


class ReplayableSequence(Sequence):
    """A sequence facade that reopens an iterator instead of retaining records."""

    def __init__(self, iterator_factory, length_factory=None):
        self._iterator_factory = iterator_factory
        self._length_factory = length_factory
        self._length = None

    def __iter__(self):
        return iter(self._iterator_factory())

    def __len__(self):
        if self._length is None:
            if self._length_factory is None:
                self._length = sum(1 for _ in self)
            else:
                self._length = int(self._length_factory())
        return self._length

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step == 1:
                return list(itertools.islice(iter(self), start, stop))
            return list(self)[index]
        try:
            index = int(index)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("sequence index must be an integer or slice") from exc
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError("algorithm stream index out of range")
        try:
            return next(itertools.islice(iter(self), index, index + 1))
        except StopIteration as exc:
            raise IndexError("algorithm stream index out of range") from exc


class ReplayableAlgorithmStream(ReplayableSequence):
    """Replay a logical stream while holding at most one decoded frame."""

    def __init__(
        self,
        run_dir,
        stream_name,
        required=False,
        representation=None,
    ):
        run_dir = pathlib.Path(run_dir)
        super().__init__(
            lambda: iter_algorithm_stream(
                run_dir,
                stream_name,
                required=required,
                representation=representation,
            ),
            length_factory=lambda: count_algorithm_stream(
                run_dir,
                stream_name,
                required=required,
                representation=representation,
            ),
        )
        self.run_dir = run_dir
        self.stream_name = str(stream_name)
        self.required = bool(required)
        self.representation = representation


def _first_payload_difference(left, right, path="$"):
    if type(left) is not type(right):
        return {
            "path": path,
            "reason": "type_mismatch",
            "jsonl_type": type(left).__name__,
            "sqlite_type": type(right).__name__,
        }
    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            missing_from_jsonl = sorted(right_keys - left_keys)
            missing_from_sqlite = sorted(left_keys - right_keys)
            return {
                "path": path,
                "reason": "key_mismatch",
                "missing_from_jsonl": missing_from_jsonl,
                "missing_from_sqlite": missing_from_sqlite,
            }
        for key in sorted(left_keys):
            difference = _first_payload_difference(
                left[key], right[key], "{}.{}".format(path, key)
            )
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {
                "path": path,
                "reason": "length_mismatch",
                "jsonl_length": len(left),
                "sqlite_length": len(right),
            }
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = _first_payload_difference(
                left_item,
                right_item,
                "{}[{}]".format(path, index),
            )
            if difference is not None:
                return difference
        return None
    if left != right:
        return {
            "path": path,
            "reason": "value_mismatch",
            "jsonl_value": left,
            "sqlite_value": right,
        }
    return None


def compare_dual_storage(run_dir, stream_names=HIGH_VOLUME_STREAMS):
    """Compare every decoded JSONL and SQLite value for a declared dual run."""

    run_dir = pathlib.Path(run_dir)
    contract = _load_storage_contract(run_dir)
    if contract["backend"] != "dual":
        raise AlgorithmFrameStoreError(
            "Dual storage comparison requires storage_backend=dual"
        )
    database_summary = inspect_sqlite_store(contract["database_path"])
    report = {
        "equivalent": database_summary["integrity_check"] == "ok",
        "database_integrity_check": database_summary["integrity_check"],
        "database_file": str(contract["database_path"]),
        "streams": {},
    }
    missing = object()
    for stream_name in stream_names:
        if stream_name not in HIGH_VOLUME_STREAMS:
            raise AlgorithmFrameStoreError(
                "Unsupported dual comparison stream: {}".format(stream_name)
            )
        jsonl_count = 0
        sqlite_count = 0
        jsonl_item_count = 0
        sqlite_item_count = 0
        first_mismatch = None
        jsonl_iterator = iter_algorithm_stream(
            run_dir,
            stream_name,
            required=True,
            representation="jsonl",
        )
        sqlite_iterator = iter_algorithm_stream(
            run_dir,
            stream_name,
            required=True,
            representation="sqlite_zlib",
        )
        for frame_index, (jsonl_payload, sqlite_payload) in enumerate(
            itertools.zip_longest(
                jsonl_iterator,
                sqlite_iterator,
                fillvalue=missing,
            )
        ):
            if jsonl_payload is not missing:
                jsonl_count += 1
                jsonl_item_count += len(
                    jsonl_payload.get(STREAM_ITEM_KEYS[stream_name], [])
                )
            if sqlite_payload is not missing:
                sqlite_count += 1
                sqlite_item_count += len(
                    sqlite_payload.get(STREAM_ITEM_KEYS[stream_name], [])
                )
            if first_mismatch is not None:
                continue
            if jsonl_payload is missing or sqlite_payload is missing:
                first_mismatch = {
                    "path": "$",
                    "reason": "frame_count_mismatch",
                    "frame_index": frame_index,
                    "jsonl_frame_present": jsonl_payload is not missing,
                    "sqlite_frame_present": sqlite_payload is not missing,
                }
                continue
            difference = _first_payload_difference(jsonl_payload, sqlite_payload)
            if difference is not None:
                difference["frame_index"] = frame_index
                first_mismatch = difference

        stream_equivalent = (
            first_mismatch is None
            and jsonl_count == sqlite_count
            and jsonl_item_count == sqlite_item_count
        )
        stream_report = {
            "equivalent": stream_equivalent,
            "frame_count": jsonl_count if stream_equivalent else None,
            "jsonl_frame_count": jsonl_count,
            "sqlite_frame_count": sqlite_count,
            "jsonl_item_count": jsonl_item_count,
            "sqlite_item_count": sqlite_item_count,
            "first_mismatch": first_mismatch,
        }
        report["streams"][stream_name] = stream_report
        if not stream_equivalent:
            report["equivalent"] = False
    return report


def trim_sqlite_stream(database_path, stream_name, keep_count):
    """Delete frames after a recording-order prefix and revalidate the database."""

    if stream_name not in HIGH_VOLUME_STREAMS:
        raise AlgorithmFrameStoreError(
            "Unsupported compressed algorithm stream: {}".format(stream_name)
        )
    keep_count = _coerce_integer(keep_count, "keep_count", required=True)
    if keep_count < 0:
        raise AlgorithmFrameStoreError("keep_count must not be negative")
    database_path = pathlib.Path(database_path)
    if not database_path.is_file():
        raise AlgorithmFrameStoreError(
            "Algorithm frame database does not exist: {}".format(database_path)
        )

    connection = None
    try:
        connection = sqlite3.connect(str(database_path))
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        cutoff_row = connection.execute(
            """
            SELECT frame_pk
              FROM stream_frames
             WHERE stream_name = ?
             ORDER BY frame_pk
             LIMIT 1 OFFSET ?
            """,
            (stream_name, keep_count),
        ).fetchone()
        if cutoff_row is not None:
            with connection:
                connection.execute(
                    "DELETE FROM stream_frames WHERE stream_name = ? AND frame_pk >= ?",
                    (stream_name, int(cutoff_row[0])),
                )
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
        summary = _summary_from_connection(connection)
        if summary["integrity_check"] != "ok":
            raise AlgorithmFrameStoreError(
                "SQLite integrity_check failed after trim: {}".format(
                    summary["integrity_check"]
                )
            )
        return summary
    except sqlite3.Error as exc:
        raise AlgorithmFrameStoreError(
            "Could not trim stream {}: {}".format(stream_name, exc)
        ) from exc
    finally:
        if connection is not None:
            connection.close()
