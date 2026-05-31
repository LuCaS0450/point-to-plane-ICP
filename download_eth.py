import argparse
import csv
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


APARTMENT_ARCHIVE = 'apartment_03-Dec-2011-18_13_33.zip'
APARTMENT_URL = (
    'https://www.research-collection.ethz.ch/server/api/core/bitstreams/'
    'fb20daee-c4da-4521-9502-34bb924654c7/content'
)
EVALUATIONS_ARCHIVE = 'evaluations.zip'
EVALUATIONS_URL = (
    'https://www.research-collection.ethz.ch/server/api/core/bitstreams/'
    'b33cd06e-d37a-4953-840d-0cbd129a7f72/content'
)
PROJECT_PAGE = 'https://doi.org/10.3929/ethz-b-000721626'


def request(url, start=0):
    headers = {'User-Agent': 'point-to-plane-icp-eth-demo/1.0'}
    if start:
        headers['Range'] = f'bytes={start}-'
    return urllib.request.Request(url, headers=headers)


def download_with_resume(url, output_path, force=False):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        print(f'skip existing {output_path}')
        return output_path
    if force and output_path.exists():
        output_path.unlink()

    part_path = output_path.with_suffix(output_path.suffix + '.part')
    start = part_path.stat().st_size if part_path.exists() else 0
    print(f'downloading {url}')
    if start:
        print(f'resuming {output_path.name} from {start / (1024 * 1024):.1f} MB')

    with urllib.request.urlopen(request(url, start=start), timeout=120) as response:
        resumed = start > 0 and getattr(response, 'status', None) == 206
        if start and not resumed:
            print('server ignored Range header; restarting download')
            start = 0
        total_header = response.headers.get('Content-Length')
        total = start + int(total_header) if total_header else None
        mode = 'ab' if resumed else 'wb'
        downloaded = start
        with part_path.open(mode) as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded * 100.0 / total
                    print(
                        f'\r  {percent:5.1f}% {downloaded / (1024 * 1024):.1f} MB',
                        end='',
                        flush=True,
                    )
        print()
    shutil.move(str(part_path), str(output_path))
    return output_path


def is_selected_apartment_file(name):
    path = PurePosixPath(name)
    parts = path.parts
    if len(parts) < 3:
        return False
    relative = PurePosixPath(*parts[1:])
    if relative.match('csv_local/Hokuyo_*.csv'):
        return True
    return str(relative) in {
        'csv_global/pose_scanner_leica.csv',
        'csv_global/overlap_apartment.csv',
    }


def is_selected_evaluation_file(name):
    path = PurePosixPath(name)
    return str(path) in {
        'evaluations/protocols/apartment_protocol.csv',
        'evaluations/validation/apartment_validation.csv',
        'evaluations/results/solution_config/Chen91_pt2plane.yaml',
        'evaluations/results/solution_config/Besl92_pt2point.yaml',
    }


def extract_selected(zip_path, output_dir, predicate, marker_name, force=False):
    output_dir = Path(output_dir)
    marker = output_dir / marker_name
    if marker.exists() and not force:
        print(f'skip extracted {zip_path.name}')
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(zip_path, 'r') as archive:
        for info in archive.infolist():
            if info.is_dir() or not predicate(info.filename):
                continue
            relative = PurePosixPath(info.filename)
            target = output_dir.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not force:
                extracted += 1
                continue
            print(f'extracting {relative}')
            with archive.open(info, 'r') as source, target.open('wb') as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            extracted += 1

    if not extracted:
        raise RuntimeError(f'no selected files found in {zip_path}')
    marker.write_text(f'{extracted} selected files extracted\n', encoding='utf-8')


def find_apartment_paths(output_dir):
    output_dir = Path(output_dir)
    roots = list(output_dir.rglob('apartment_03-Dec-2011-18_13_33'))
    scene_roots = [root for root in roots if root.is_dir()]
    if not scene_roots:
        raise FileNotFoundError(f'Apartment scene directory not found under {output_dir}')
    scene_root = sorted(scene_roots, key=lambda path: (len(path.parts), str(path)))[0]

    protocol = output_dir / 'evaluations' / 'protocols' / 'apartment_protocol.csv'
    validation = output_dir / 'evaluations' / 'validation' / 'apartment_validation.csv'
    return {
        'root': scene_root,
        'scan_dir': scene_root / 'csv_local',
        'pose_csv': scene_root / 'csv_global' / 'pose_scanner_leica.csv',
        'overlap_csv': scene_root / 'csv_global' / 'overlap_apartment.csv',
        'protocol_csv': protocol,
        'validation_csv': validation,
    }


def validate_dataset(output_dir):
    try:
        paths = find_apartment_paths(output_dir)
    except FileNotFoundError as exc:
        print(f'dataset validation failed: {exc}')
        return False

    missing = [
        str(path)
        for key, path in paths.items()
        if key != 'root' and key != 'scan_dir' and not path.exists()
    ]
    scan_paths = sorted(paths['scan_dir'].glob('Hokuyo_*.csv'))
    if not scan_paths:
        missing.append(f"{paths['scan_dir']}: no Hokuyo_*.csv files")

    referenced = set()
    if paths['protocol_csv'].exists():
        with paths['protocol_csv'].open(newline='', encoding='utf-8-sig') as handle:
            for row in csv.DictReader(handle, skipinitialspace=True):
                referenced.add(row['reference'].strip())
                referenced.add(row['reading'].strip())
    scan_names = {path.name for path in scan_paths}
    missing_scans = sorted(referenced - scan_names)
    if missing_scans:
        missing.append(f'missing scans referenced by protocol: {", ".join(missing_scans)}')

    if missing:
        print('dataset validation found missing files:')
        for item in missing:
            print(f'  - {item}')
        return False

    print(f'dataset validation passed: {len(scan_paths)} scans, {len(referenced)} referenced scans')
    for key, path in paths.items():
        print(f'  {key}: {path}')
    return True


def main():
    parser = argparse.ArgumentParser(description='Download selected ETH Apartment registration files.')
    parser.add_argument('--output', default='data/eth', help='dataset output directory')
    parser.add_argument('--force', action='store_true', help='redownload and re-extract files')
    parser.add_argument('--keep-archives', action='store_true', help='keep downloaded zip files')
    parser.add_argument('--validate-only', action='store_true', help='validate an existing extracted dataset')
    parser.add_argument('--list-urls', action='store_true', help='print official URLs without downloading')
    args = parser.parse_args()

    output_dir = Path(args.output)
    if args.list_urls:
        print(f'project: {PROJECT_PAGE}')
        print(f'apartment: {APARTMENT_URL}')
        print(f'evaluations: {EVALUATIONS_URL}')
        return 0
    if args.validate_only:
        return 0 if validate_dataset(output_dir) else 1

    archive_dir = output_dir / '_archives'
    try:
        apartment_zip = download_with_resume(
            APARTMENT_URL, archive_dir / APARTMENT_ARCHIVE, force=args.force
        )
        evaluations_zip = download_with_resume(
            EVALUATIONS_URL, archive_dir / EVALUATIONS_ARCHIVE, force=args.force
        )
        extract_selected(
            apartment_zip,
            output_dir,
            is_selected_apartment_file,
            '.apartment-selected.extracted',
            force=args.force,
        )
        extract_selected(
            evaluations_zip,
            output_dir,
            is_selected_evaluation_file,
            '.evaluations-selected.extracted',
            force=args.force,
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, zipfile.BadZipFile) as exc:
        print(f'download failed: {exc}', file=sys.stderr)
        return 1

    ok = validate_dataset(output_dir)
    if ok and not args.keep_archives and archive_dir.exists():
        shutil.rmtree(archive_dir)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
