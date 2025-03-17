import os
import re
import time
import json
import hashlib
import ffmpeg
import mimetypes
import shutil
import logging
import base64
from dataclasses import dataclass
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from PIL import Image, ExifTags
from PIL.TiffImagePlugin import IFDRational


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

logger = get_logger()


###############################################################################
#                               FileDescriptor                                #
###############################################################################

def exif_encoder(obj):
    if isinstance(obj, IFDRational):
        return float(obj)
    elif isinstance(obj, bytes):
        return base64.b64encode(obj).decode("utf-8")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def file_type(path: Path | str) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        return "other"
    if mime_type.startswith("image"):
        return "image"
    elif mime_type.startswith("video"):
        return "video"
    else:
        return "other"


def image_metadata(path: Path | str):
    with Image.open(path) as img:
        meta = {}
        for k, v in img.getexif().items():
            if k in ExifTags.TAGS:
                meta[ExifTags.TAGS[k]] = v
    return meta


def video_metadata(path: Path | str):
    return ffmpeg.probe(path)


def datetime_from_path(path: str):
    # covers:
    # /path/to/2010/05/image.jpg
    # /path/to/2010/subdir/image.jpg
    matched = re.search(r".*/(19\d{2}|20\d{2})/(\d{2}/)?", path)
    if matched:
        year_str, month_str = matched.groups()
        year = int(year_str)
        month = int(month_str.strip("/")) if month_str else 1
        return datetime(year, month, 1)
    return None


@dataclass
class FileDescriptor:
    path: str
    size: int
    hash: str
    typ: str
    _meta: dict | None = None

    @staticmethod
    def from_file(path: Path | str):
        path = str(path)
        with open(path, "rb") as fp:
            content = fp.read()
        size = len(content)
        hash = hashlib.sha1(content).hexdigest()
        typ = file_type(path)
        return FileDescriptor(path=path, size=size, hash=hash, typ=typ)

    @staticmethod
    def from_dict(dct: dict):
        return FileDescriptor(**dct)

    def __repr__(self):
        return f"FileDescriptor(path='{self.path}', size={self.size})"

    @property
    def name(self):
        return os.path.basename(self.path)

    @property
    def meta(self):
        if self._meta is None:
            if self.typ == "image":
                self._meta = image_metadata(self.path)
            elif self.typ == "video":
                self._meta = video_metadata(self.path)
            # otherwise: keep _meta == None
        return self._meta

    def is_same(self, other):
        self_dt = self.record_time or datetime(1, 1, 1)
        other_dt = other.record_time or datetime(1, 1, 1)
        return self.hash == other.hash and self_dt.year == other_dt.year and self_dt.month == other_dt.month

    @property
    def record_time(self):
        # empty files have no metadata, so we give up on them
        if self.size == 0:
            return None
        dt = None
        # try to extract datetime from metadata
        if self.typ == "image":
            dts = self.meta.get("DateTimeOriginal") or self.meta.get("DateTime")
            dt = datetime.strptime(dts, "%Y:%m:%d %H:%M:%S") if dts else None
        elif self.typ == "video":
            from_tag = self.meta.get("format", {}).get("tags", {}).get("creation_time")
            if from_tag and not from_tag.startswith("1970"):
                dt = datetime.strptime(from_tag, "%Y-%m-%dT%H:%M:%S.%fZ")
        # as a last resort, try to extract datetime from path
        if dt is None:
            dt = datetime_from_path(self.path)
        return dt

    @property
    def album(self):
        album = os.path.basename(os.path.dirname(self.path))
        if not re.match(r"^[0-9\-\._ ]+$", album):
            # doesn't look like a date
            return album
        return None


###############################################################################
#                                    Index                                    #
###############################################################################

class Index:
    def __init__(self, cachefile: str | None = None, recreate=False):
        self.cachefile = cachefile or os.path.abspath("media-index.jsonl")
        if recreate and os.path.exists(self.cachefile):
            os.remove(self.cachefile)
        self.items = []
        self.hashes = defaultdict(lambda: [])    # hash -> [fd]
        self.paths = defaultdict(lambda: [])     # path -> [fd]
        self.update_from_cache(self.cachefile)

    def __repr__(self):
        return f"Index(cachefile='{self.cachefile}', len={len(self.items)})"

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def update_from_cache(self, cachefile: str):
        logger.info(f"Pre-loading index from {cachefile}")
        if os.path.exists(cachefile):
            with open(cachefile) as fp:
                lines = fp.readlines()
            for line in tqdm(lines):
                dct = json.loads(line)
                fd = FileDescriptor.from_dict(dct)
                self.add(fd)

    def add(self, fd: FileDescriptor):
        existing = self.hashes[fd.hash]
        if existing and any(fd.path == e.path for e in existing):
            # file already added to the index
            return False
        self.items.append(fd)
        self.hashes[fd.hash].append(fd)
        self.paths[fd.path].append(fd)
        try:
            with open(self.cachefile, "a") as fp:
                fp.write(json.dumps(fd.__dict__, default=exif_encoder) + "\n")
        except Exception as ex:
            logger.warning(f"Failed to write {fd} to the cache because of {repr(ex)}")
        return True

    def update(self, root: Path | str):
        root = Path(root).expanduser()
        pbar = tqdm(sorted(root.rglob("*")))
        for path in pbar:
            if path.is_file() and str(path) not in self.paths:
                fd = FileDescriptor.from_file(path)
                self.add(fd)
            relpath = str(path.relative_to(root))
            relpath = relpath.rjust(30)[-30:]
            pbar.set_description(relpath)


def test_index():
    self = Index(recreate=True)
    fd = FileDescriptor.from_file('/Users/az/ElementsBackup/GooglePhotos/photos/2016/04/IMG_3770.JPG')
    assert self.add(fd) == True
    assert self.add(fd) == False
    self.update("~/ElementsBackup")

###############################################################################
#                                 reorganize                                  #
###############################################################################

def maybe_increment_path(path: str):
    if not os.path.exists(path):
        # already ok
        return path
    dirname, basename = os.path.split(path)
    name, ext = os.path.splitext(basename)
    match = re.match(r'^(.*?)\s*\((\d+)\)$', name)
    if match:
        base = match.group(1)
        index = int(match.group(2))
        new_name = f"{base} ({index + 1})"
    else:
        new_name = f"{name} (1)"
    new_path = os.path.join(dirname, new_name + ext)
    return new_path


def copy_with_retry(src, dst, n_retries=10):
    success = False
    while n_retries > 0 and not success:
        try:
            shutil.copy(src, dst)
            success = True
        except:
            print(f"Failed to copy: {src} -> {dst}. Retrying in 3 seconds...")
            time.sleep(3)
            n_retries -= 1
    if not success:
        raise ValueError(f"Failed to copy {src} -> {dst} after 10 retries")



def output_path(fd: FileDescriptor, dst: str):
    """
    Generate (base) output path given file descriptor and destination root.
    """
    dt = fd.record_time
    if dt:
        album_or_month = fd.album or str(dt.month)
        base_out_path = f"{dst}/{dt.year}/{album_or_month}/{fd.name}"
    else:
        maybe_album = fd.album + "/" if fd.album else ""
        base_out_path = f"{dst}/(no-date)/{maybe_album}/{fd.name}"
    return base_out_path

FD = None

def reorganize(src: str | list[str], dest: str):
    if isinstance(src, str) or isinstance(src, Path):
        src = [src]
    print("Indexing source")
    index = Index()
    for root in src:
        index.update(root)
    print("Indexing destination")
    dest_index = Index(cachefile=os.path.abspath("dest-media-index.jsonl"))
    dest_index.update(dest)
    print(f"Files already in destination: {len([fd for fd in index if fd.hash in dest_index.hashes])}")
    print(f"Copying to the {dest}")
    dest = os.path.expanduser(dest).rstrip("/")
    for fd in tqdm(index):
        global FD
        FD = fd
        # if file is invalid or not a media, then skip
        if fd.size == 0 or (fd.typ != "image" and fd.typ != "video"):
            continue
        # if there's already such a file in the destination, then skip
        dest_fds = dest_index.hashes[fd.hash]
        if any(fd.is_same(dest_fd) for dest_fd in dest_fds):
            continue
        # otherwise, copy file
        base_out_path = output_path(fd, dest)
        out_path = maybe_increment_path(base_out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        copy_with_retry(fd.path, out_path)
        dest_fd = deepcopy(fd)
        dest_fd.path = out_path
        dest_index.add(dest_fd)
    print("Done!")


def main():
    src = "~/ElementsBackup"
    dest = "/Volumes/Elements/photos"
    # src = "~/Takeout"
    # dest = "~/TakeoutReorganized"
    reorganize(src, dest)