import os
import requests
from tqdm import tqdm

def download_model(url: str) -> None:
    file_name = os.path.join(*url.split('/')[-2:])
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    response = requests.get(url, stream=True)
    with open(file_path, 'wb') as f:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=file_name, total=int(response.headers.get('content-length', 0))) as pbar:
            for chunk in response.iter_content(chunk_size=4096):
                f.write(chunk)
                pbar.update(len(chunk))

if __name__ == '__main__':
    download_model('https://github.com/HolyWu/vs-rife/releases/download/model18/contextnet.pkl')
    download_model('https://github.com/HolyWu/vs-rife/releases/download/model18/flownet.pkl')
    download_model('https://github.com/HolyWu/vs-rife/releases/download/model18/unet.pkl')

    download_model('https://github.com/HolyWu/vs-rife/releases/download/model23/contextnet.pkl')
    download_model('https://github.com/HolyWu/vs-rife/releases/download/model23/flownet.pkl')
    download_model('https://github.com/HolyWu/vs-rife/releases/download/model23/unet.pkl')

    download_model('https://github.com/HolyWu/vs-rife/releases/download/model24/contextnet.pkl')
    download_model('https://github.com/HolyWu/vs-rife/releases/download/model24/flownet.pkl')
    download_model('https://github.com/HolyWu/vs-rife/releases/download/model24/unet.pkl')

    download_model('https://github.com/HolyWu/vs-rife/releases/download/model31/contextnet.pkl')
    download_model('https://github.com/HolyWu/vs-rife/releases/download/model31/flownet.pkl')
    download_model('https://github.com/HolyWu/vs-rife/releases/download/model31/unet.pkl')

    download_model('https://github.com/HolyWu/vs-rife/releases/download/model35/flownet.pkl')

    download_model('https://github.com/HolyWu/vs-rife/releases/download/model38/flownet.pkl')
