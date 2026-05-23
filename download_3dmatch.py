import argparse
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


PROJECT_PAGE = 'https://3dmatch.cs.princeton.edu/'
DOWNLOAD_BASE = 'http://3dvision.princeton.edu/projects/2016/3DMatch/downloads/'

SCENES = [
    '7-scenes-redkitchen',
    'sun3d-home_at-home_at_scan1_2013_jan_1',
    'sun3d-home_md-home_md_scan9_2012_sep_30',
    'sun3d-hotel_uc-scan3',
    'sun3d-hotel_umd-maryland_hotel1',
    'sun3d-hotel_umd-maryland_hotel3',
    'sun3d-mit_76_studyroom-76-1studyroom2',
    'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika',
]


def request(url):
    return urllib.request.Request(url, headers={'User-Agent': 'point-to-plane-icp-demo/1.0'})


def read_project_page():
    try:
        with urllib.request.urlopen(request(PROJECT_PAGE), timeout=30) as response:
            return response.read().decode('utf-8', errors='ignore')
    except Exception:
        return ''


def normalize_url(href):
    return urllib.parse.urljoin(PROJECT_PAGE, href.replace('&amp;', '&'))


def scrape_zip_urls():
    html = read_project_page()
    urls = [normalize_url(href) for href in re.findall(r'href=["\']([^"\']+\.zip)["\']', html)]
    by_scene = {scene: {'fragments': [], 'evaluation': []} for scene in SCENES}
    for url in urls:
        for scene in SCENES:
            if scene in url:
                if 'evaluation' in url:
                    by_scene[scene]['evaluation'].append(url)
                elif 'fragment' in url:
                    by_scene[scene]['fragments'].append(url)
    return by_scene


def fallback_urls(scene, kind):
    if kind == 'fragments':
        paths = [
            f'fragments/{scene}.zip',
            f'fragments/{scene}-fragments.zip',
        ]
    else:
        paths = [
            f'evaluation-files/{scene}-evaluation.zip',
            f'evaluation-files/{scene}.zip',
            f'evaluation/{scene}-evaluation.zip',
            f'evaluation/{scene}.zip',
        ]
    return [urllib.parse.urljoin(DOWNLOAD_BASE, path) for path in paths]


def unique(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def download_with_candidates(candidates, output_path, force=False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        print(f'skip existing {output_path}')
        return output_path

    last_error = None
    tmp_path = output_path.with_suffix(output_path.suffix + '.part')
    for url in candidates:
        print(f'trying {url}')
        try:
            with urllib.request.urlopen(request(url), timeout=60) as response:
                total = response.headers.get('Content-Length')
                total = int(total) if total else None
                downloaded = 0
                with tmp_path.open('wb') as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            percent = downloaded * 100.0 / total
                            print(f'\r  {percent:5.1f}% {downloaded / (1024 * 1024):.1f} MB', end='')
                print()
            shutil.move(str(tmp_path), str(output_path))
            return output_path
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            print(f'  failed: {exc}')
    raise RuntimeError(f'could not download {output_path.name}: {last_error}')


def extract_zip(zip_path, output_dir, force=False):
    marker = output_dir / (zip_path.stem + '.extracted')
    if marker.exists() and not force:
        print(f'skip extracted {zip_path.name}')
        return

    print(f'extracting {zip_path.name}')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)
    marker.write_text('ok\n')


def validate_dataset(output_dir):
    missing = []
    for scene in SCENES:
        ply_files = list(output_dir.rglob(f'{scene}/cloud_bin_*.ply'))
        gt_logs = [path for path in output_dir.rglob('gt.log') if scene in str(path)]
        if not ply_files:
            missing.append(f'{scene}: no cloud_bin_*.ply files')
        if not gt_logs:
            missing.append(f'{scene}: no gt.log')
    if missing:
        print('dataset validation found missing files:')
        for item in missing:
            print(f'  - {item}')
        return False
    print('dataset validation passed')
    return True


def main():
    parser = argparse.ArgumentParser(description='Download the official 3DMatch geometric registration benchmark.')
    parser.add_argument('--output', default='data/3dmatch', help='dataset output directory')
    parser.add_argument('--keep-zips', action='store_true', help='keep downloaded zip files after extraction')
    parser.add_argument('--force', action='store_true', help='redownload and re-extract files')
    parser.add_argument('--list-urls', action='store_true', help='print candidate URLs without downloading')
    args = parser.parse_args()

    output_dir = Path(args.output)
    archive_dir = output_dir / '_archives'
    scraped = scrape_zip_urls()

    for scene in SCENES:
        for kind in ('fragments', 'evaluation'):
            candidates = unique(scraped.get(scene, {}).get(kind, []) + fallback_urls(scene, kind))
            zip_path = archive_dir / f'{scene}-{kind}.zip'
            if args.list_urls:
                print(f'{scene} {kind}:')
                for candidate in candidates:
                    print(f'  {candidate}')
                continue
            downloaded = download_with_candidates(candidates, zip_path, force=args.force)
            extract_zip(downloaded, output_dir, force=args.force)

    if args.list_urls:
        return 0

    ok = validate_dataset(output_dir)
    if not args.keep_zips and archive_dir.exists():
        shutil.rmtree(archive_dir)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
