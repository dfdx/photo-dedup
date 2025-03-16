import os
import re
import hashlib
import ffmpeg
import mimetypes
import shutil
from pathlib import Path
from datetime import datetime
from itertools import islice
from tqdm import tqdm
from PIL import Image, ExifTags


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



class FileDescriptor:
    def __init__(self, path: Path | str):
        self.path = str(path)
        with open(path, "rb") as fp:
            content = fp.read()
        self.size = len(content)
        self.hash = hashlib.sha1(content).hexdigest()
        self.typ = file_type(path)
        self._image_meta = None
        self._video_meta = None

    def __repr__(self):
        return f"FileDescriptor(path='{self.path}', size={self.size})"

    @property
    def name(self):
        return os.path.basename(self.path)

    @property
    def image_meta(self):
        if self.typ != "image":
            return None
        if self.size == 0:
            self._image_meta = {}
        if self._image_meta is None:
            self._image_meta = image_metadata(self.path)
        return self._image_meta

    @property
    def video_meta(self):
        if self.typ != "video":
            return None
        if self.size == 0:
            self._video_meta = {}
        if self._video_meta is None:
            self._video_meta = video_metadata(self.path)
        return self._video_meta

    @property
    def meta(self):
        if self.typ == "image":
            return self.image_meta
        elif self.typ == "video":
            return self.video_meta
        else:
            return None

    @property
    def record_time(self):
        # time from path has highest priority since somebody has already set it manually
        dt = datetime_from_path(self.path)
        if dt:
            return dt
        # empty files have no metadata, so we give up on them
        if self.size == 0:
            return None
        # try to extract datetime from metadata
        if self.typ == "image":
            dts = self.meta.get("DateTimeOriginal") or self.meta.get("DateTime")
            return datetime.strptime(dts, "%Y:%m:%d %H:%M:%S") if dts else None
        elif self.typ == "video":
            from_tag = self.meta.get("format", {}).get("tags", {}).get("creation_time")
            if from_tag and not from_tag.startswith("1970"):
                return datetime.strptime(from_tag, "%Y-%m-%dT%H:%M:%S.%fZ")
            return None
        return None

    @property
    def album(self):
        # matches /path/to/2010/album_name/image.jpg
        album_matched = re.search(r"/(19\d{2}|20\d{2})/(.*)/.*\..*$", self.path)
        month_matched = re.search(r"/(19\d{2}|20\d{2})/(\d{2})/.*\..*$", self.path)
        if album_matched and not month_matched:
            return album_matched.groups()[-1]
        return None


def _same_file(fd1: FileDescriptor, fd2: FileDescriptor):
    return fd1.hash == fd2.hash and fd1.record_time == fd2.record_time


def find_issues(index: list[FileDescriptor]):
    hashes = {}
    collisions = []
    duplicates = []
    empty = []
    for fd in tqdm(index):
        if fd.size == 0:
            empty.append(fd)
            continue
        if fd.hash in hashes:
            existing_fd = hashes[fd.hash]
            if _same_file(existing_fd, fd):
                duplicates.append((existing_fd, fd))
            else:
                collisions.append((existing_fd, fd))
        else:
            hashes[fd.hash] = fd
    return {
        "collisions": collisions,
        "duplicates": duplicates,
        "empty": empty,
    }


def build_index(root: Path | str):
    index = []
    pbar = tqdm(sorted(root.rglob("*")))
    for path in pbar:
        if path.is_file():
            fd = FileDescriptor(path)
            index.append(fd)
        relpath = str(path.relative_to(root))
        relpath = relpath.rjust(30)[-30:]
        pbar.set_description(relpath)
    return index


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


def reorganize(src: str | list[str], dest: str):
    if isinstance(src, str) or isinstance(src, Path):
        src = [src]
    print("Indexing files")
    index = []
    for root in src:
        root = Path(root).expanduser()
        idx = build_index(root)
        index.extend(idx)
    print("Analyzing for issues")
    issues = find_issues(index)
    duplicates = set(fd.path for _, fd in issues["duplicates"])
    collisions = set(fd.path for _, fd in issues["collisions"])
    print(f"Found {len(duplicates)} duplicates and {len(collisions)} collisions")
    print(f"Copying to the {dest}")
    dest = os.path.expanduser(dest).rstrip("/")
    for fd in tqdm(index):
        if fd.size == 0:
            continue
        if fd.path in duplicates:
            continue
        if fd.typ != "image" and fd.typ != "video":
            continue
        dt = fd.record_time
        if dt:
            album_or_month = fd.album or str(dt.month)
            base_out_path = f"{dest}/{dt.year}/{album_or_month}/{fd.name}"
        else:
            base_out_path = f"{dest}/(no-date)/{fd.name}"
        out_path = maybe_increment_path(base_out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        shutil.copy(fd.path, out_path)
    print("Done!")


def main():
    src = "~/ElementsBackup"
    # dest = "/Volumes/Elements/photos/"
    dest = "~/ElementsTestOut"

    index = build_index(src)
    issues = find_issues(index)
