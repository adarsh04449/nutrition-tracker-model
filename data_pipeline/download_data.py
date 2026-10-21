"""Download real overhead RGB photos only. Default: a small verification subset."""
import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

BASE = 'https://storage.googleapis.com/nutrition5k_dataset/'
PREFIX = 'nutrition5k_dataset/'


def fetch(url, path):
    if path.exists():
        return False
    tmp = path.with_suffix(path.suffix + '.part')
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=90) as response:
        with tmp.open('wb') as f:
            shutil.copyfileobj(response, f)
    tmp.replace(path)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limit', type=int, default=12, help='Number of RGB photos; 0 downloads all')
    p.add_argument('--root', type=Path, default=Path('data/raw'))
    p.add_argument('--workers', type=int, default=8, help='Concurrent image downloads')
    a = p.parse_args()
    if a.limit < 0:
        p.error('limit must be nonnegative')
    if a.workers < 1:
        p.error('workers must be positive')
    for name in ('dish_metadata_cafe1.csv', 'dish_metadata_cafe2.csv'):
        rel = 'metadata/' + name
        fetch(BASE + PREFIX + rel, a.root / rel)
    names, token = [], None
    while True:
        params = {'prefix': PREFIX + 'imagery/realsense_overhead/', 'maxResults': 1000}
        if token:
            params['pageToken'] = token
        with urlopen('https://storage.googleapis.com/storage/v1/b/nutrition5k_dataset/o?' + urlencode(params), timeout=90) as r:
            page = json.load(r)
        for item in page.get('items', []):
            name = item['name']
            if not name.endswith('/rgb.png'):
                continue
            names.append(name)
            if a.limit and len(names) >= a.limit:
                break
        if a.limit and len(names) >= a.limit:
            break
        token = page.get('nextPageToken')
        if not token:
            break

    def download(name):
        return fetch(BASE + name, a.root / name.removeprefix(PREFIX))

    downloaded = 0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for count, (name, created) in enumerate(zip(names, pool.map(download, names)), 1):
            downloaded += created
            print(f'{count}/{len(names)}: {name}', flush=True)
    print(f'Finished: {len(names)} overhead RGB photos ({downloaded} downloaded, '
          f'{len(names) - downloaded} already present)')


if __name__ == '__main__':
    main()
