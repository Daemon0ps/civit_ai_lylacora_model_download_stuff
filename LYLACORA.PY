from __future__ import annotations
import os
import re
import cv2
import ssl
import sys
import json
import nltk
import torch
import base64
import shutil
import string
import urllib
import hashlib
import keyring
import sqlite3
import requests
import xlsxwriter
import numpy as np
import pandas as pd
import urllib.request
from enum import Enum
from tqdm import tqdm
from io import BytesIO
from time import sleep
from random import randint
import matplotlib.pyplot as plt
from PIL.ExifTags import TAGS
from iptcinfo3 import IPTCInfo
from matplotlib import gridspec
from unidecode import unidecode
from dataclasses import dataclass
from dateutil.parser import parse
from nltk.corpus import stopwords
from types import SimpleNamespace
from datetime import datetime, timezone
from nltk.tokenize import word_tokenize
from requests_oauthlib import OAuth1Session
from urllib.error import HTTPError, URLError
from requests import Response, ConnectionError
from IPython.display import display,clear_output
from requests.adapters import HTTPAdapter, Retry
from traceback_with_variables import activate_by_import
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import singledispatch, update_wrapper,singledispatchmethod
from PIL import Image,ImageFile,ImageOps,ImageFilter,ImageChops,ImageDraw,ImageDraw2,PngImagePlugin
Image.MAX_IMAGE_PIXELS = None
ImageDraw2.Font
ImageFile.LOAD_TRUNCATED_IMAGES = True
nltk.download("punkt",quiet=True)
nltk.download("stopwords",quiet=True)
nltk.download("tagsets",quiet=True)
sw = set(stopwords.words("english"))

@dataclass(frozen=False)
class c:
    api_token = keyring.get_password("civitai","api_token")
    lora_dir = "/mnt/z/sd/Lora/"
    lyco_dir = "/mnt/z/sd/LyCORIS/"
    sd_dir = "/mnt/z/sd/Stable-diffusion/"
    save_path = "f:/sd/civitai/"
    url0:str = "https://civitai.com"
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=5)
    adapter = HTTPAdapter(max_retries=30)
    cxn_state:bool = False
    headers = {
        'Authorization': f"Bearer {api_token}",
        "Accept" : "application/json+canvas-string-ids",}
    url0 = "https://civitai.com"
    api_ep = {
        "creators":"/api/v1/creators",
        "imgs":"/api/v1/images",
        "models":"/api/v1/models",
        "model_id":"/api/v1/models",
        "model_ver_id":"/api/v1/model-versions/",
        "model_hash":"/api/v1/model-versions/by-hash",
        "tags":"/api/v1/tags"
    }
    lof = []
    lyf = []
    lylacora = []
    lora_files = []
    lyco_files = []
    lylacora_files = []
    _model_id_ep = lambda x: str(f'https://civitai.com/api/v1/models/{x}')
    _model_vers_id_ep = lambda x: str(f'https://civitai.com/api/v1/model-versions/{x}')
    _model_by_hash_ep = lambda x: str(f'https://civitai.com/api/v1/model-versions/by-hash/{x}')
    _ld = lambda x: { c.ld[k]:v for k, v in dict(x).items() }
    _lh = lambda x: list([dict(x)[k]['url'] for k, v in dict(x).items()])
    now = lambda: datetime.strftime(datetime.now(), r"%Y%m%d%H%M%S")
    _hex_chk = lambda x: str(''.join(l for l in [x for x in x.replace(chr(32),chr(95))] if l in set('ABCDEF0123456789')))
    fn_rm = lambda s: str(''.join(l for l in [x for x in s.replace(chr(32),chr(95))] if l in set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.')))
    np_k2_ord_srt = lambda npl: [ 
                        x for x in [np.array(c)[0:].reshape(-1) for c in [ np.array(f[np.argsort(g)]).astype('str').tolist()
                        for f,g in [np.unique(np.array([[str(x[0]).split(chr(32))
                        for x in sorted([[s,npl.count(s)] 
                        for s in [ np.array(y[np.argsort(z)]).astype('str').tolist()[0]
                        for y,z in [np.unique(np.array(npl),return_index=True)]]],
                        key=lambda x: x[1], reverse=False)]]),return_index=True)]]]]
    rm_sp = lambda s: str(re.sub('\s+', ' ', (re.sub('_', '', (re.sub('[^a-zA-z0-9\s]', '', unidecode(s)))))).strip().lower())
    rm_aw = lambda s: "".join(s.split())
    rm_sw = lambda s: str(' '.join(w for w in word_tokenize(s) if w not in sw))
    rm_wh = lambda s: " ".join(s.split())
    rm_pu = lambda s: s.translate(str.maketrans('', '', string.punctuation))


    
    def __post_init__(self):
        self.ep = c.ep
        self._ld = c._ld
        self._lh = c._lh
        self.now = c.now
        self.lof = c.lof 
        self.lyf = c.lyf 
        self.tday = c.tday
        self.tstr = c.tstr
        self.url0 = c.url0
        self.token = c.token
        self.sd_dir = c.sd_dir 
        self.headers = c.headers
        self.lora_dir = c.lora_dir 
        self.lyco_dir = c.lyco_dir 
        self.lylacora = c.lylacora 
        self.cxn_state = c.cxn_state
        self.save_path = c.save_path
        self.api_token = c.api_token 
        self._model_id = c._model_id_ep
        self.lora_files = c.lora_files 
        self.lyco_files = c.lyco_files 
        self.lylacora_files = c.lylacora_files 
        self._model_by_hash = c._model_by_hash_ep
        self._model_vers_id = c._model_vers_id_ep
        super().__setattr__("attr_name", self)
        
        c.lof = [
            c.lora_dir+f for f 
            in os.listdir(c.lora_dir[:-1:]) 
            if os.path.isfile(c.lora_dir+f) 
            and f[-(f[::-1].find('.')):].lower() in ['ckpt','pt','safetensors']]
        
        c.lyf = [
            c.lyco_dir+f for f 
            in os.listdir(c.lyco_dir[:-1:]) 
            if os.path.isfile(c.lyco_dir+f) 
            and f[-(f[::-1].find('.')):].lower() in ['ckpt','pt','safetensors']]

    @staticmethod
    def _cxn()->requests.Session:
        if c.cxn_state:
            try:
                c.session.close()
                c.cxn_state = False
            except Exception as cxe:
                pass
        c.session = requests.Session()
        c.retry = Retry(connect=3, backoff_factor=5)
        c.adapter = HTTPAdapter(max_retries=30)
        c.session.mount('https://', c.adapter)
        c.cxn_state = True
        return c.session

    @staticmethod
    def dns(d:dict) -> SimpleNamespace:
        return SimpleNamespace(**d)

    @staticmethod
    def _get(ep:str)->Response:
        resp:Response = None
        try:
            c.session = c._cxn()
            resp = c.session.get(ep, headers=c.headers)
            c.session.close()
            c.cxn_state = False
            return resp
        except KeyboardInterrupt:
            sys.exit()
        except ConnectionError as ce:
            print(ce)
            pass
        except Exception as e:
            print(e)
            pass

    @staticmethod
    def _dl(ep:str)->Response:

        def chk_h(file:str)->str:
            with open(file,"rb") as fi:
                file_bytes = fi.read()
            return str(hashlib.sha256(file_bytes).hexdigest())
        _r:Response.json = None
        _d:str = None
        _f_r:Response = None
        _r_h:str = None
        try:
            c.session = c._cxn()
            _r = c._get(c._model_id_ep(ep)).json()
            _r_fl = _r['modelVersions'][0]['files']
            for _fl in _r_fl:
                _d:str = None
                _r_h:str = None
                _f_r:Response = None
                _f_r_b:BytesIO = None
                _f_r_b_h:str = None
                _d = _fl['downloadUrl']
                _r_h = _fl['hashes']['SHA256']
                print(_r_h)
                print(_d)
                _f_r = c._get(_d)
                print(c.lora_dir+c.fn_rm(str(_f_r.headers['Content-Disposition'].split('"')[1])))
                with open(c.lora_dir+c.fn_rm(str(_f_r.headers['Content-Disposition'].split('"')[1])),"wb")as fi:
                    _f_r_b = BytesIO(_f_r.content)
                    fi.write(_f_r_b.read())
                    _f_r_b_h = chk_h(c.lora_dir+c.fn_rm(str(_f_r.headers['Content-Disposition'].split('"')[1])))
                    _f_r_b_h = c._hex_chk(_f_r_b_h.strip().upper())
                    _r_h = c._hex_chk(_r_h.strip().upper())
                    print("[ ! ] SHA256 HASH CHECK [ ! ]")
                    print("Expected: ",c._hex_chk(_r_h.strip().upper()))
                    print("Received: ",c._hex_chk(_f_r_b_h.strip().upper()))
                    if str(_f_r_b_h) == str(_r_h):
                        print("( ^_^ ) HASH CHECK OK ( ^_^ )")
                    elif str(_f_r_b_h) != str(_r_h):
                        print("[ ! ] HASH CHECK **FAILED** DAME DESU [ ! ]")
                        print("あなたのモデルはヘルペスを患っています。")
            c.session.close()
            c.cxn_state = False
        except KeyboardInterrupt:
            sys.exit()
        except ConnectionError as ce:
            print(ce)
            return None
        except Exception as e:
            print(e)
            return None


    def _hash_check(ep:str):
        try:
            c.session = c._cxn()
            _r = c._get(c._model_id_ep(ep)).json()
            _r_fl = _r['modelVersions'][0]['files']
            for _fl in _r_fl:
                _d:str = None
                _r_h:str = None
                _d = _fl['downloadUrl']
                _r_h = _fl['hashes']['SHA256']
                print(_r_h)
        except Exception as e:
            print(e)
            pass


    @staticmethod
    def _get_model_info_by_hash(hash:str):
        try:
            resp = c._get(f"{c.url0}{c.api_ep['model_hash']}{hash}")
            return (True, json.loads(resp))
        except HTTPError as e_err:
            return (False, e_err)
        except URLError as u_err:
            return (False, u_err)


def statbar(tot: int, desc: str)->tqdm:
    l_bar = "{desc}: {percentage:3.0f}%|"
    r_bar = "| {n_fmt}/{total_fmt} [elapsed: {elapsed} / remaining: {remaining}] "
    bar = "{rate_fmt}{postfix}]"
    status_bar = tqdm(total=tot, desc=desc, bar_format=f"{l_bar}{bar}{r_bar}")
    return status_bar

def f_split(f: str) -> list:#0:"x:/basedir/",1:"filename",2:"ext"
    return list([f[:len(f)-(f[::-1].find(chr(47))):],
                f[len(f)-(f[::-1].find(chr(47))):(len(f))-1-len(f[-(f[::-1].find(".")):])],
                f[-(f[::-1].find(".")):],]
                for f in[f.replace(chr(92),chr(47))])[0]

def get_hash(file:str,t:bool)->None:
    with open(file,"rb") as fi:
        file_bytes = fi.read()
    h = str(hashlib.sha256(file_bytes).hexdigest())
    c.lora_files.append([file,h]) if t else c.lyco_files.append([file,h])
    return 

def get_model_info_by_hash(hash:str):
    try:
        resp = c._get(f"{c.url0}{c.api_ep['model_hash']}{hash}")
        return (True, json.loads(resp))
    except HTTPError as e_err:
        return (False, e_err)
    except URLError as u_err:
        return (False, u_err)

def load_lylocora_files()->None:
    with ThreadPoolExecutor(8) as executor:
        status_bar = statbar(int(len(c.lof)),"hashing loras")
        futures = [executor.submit(get_hash, file, True) for file in c.lof]
        for _ in as_completed(futures):
            status_bar.update(n=1)
        status_bar.close()
    with ThreadPoolExecutor(8) as executor:
        status_bar = statbar(int(len(c.lyf)),"hashing loras")
        futures = [executor.submit(get_hash, file, False) for file in c.lyf]
        for _ in as_completed(futures):
            status_bar.update(n=1)
        status_bar.close()
    with open(c.save_path+"lora_list.txt","wt") as fi:
        for x in c.lora_files:
            fi.write(fr"{x[0]},{x[1]}{chr(10)}")
    with open(c.save_path+"lyco_list.txt","wt") as fi:
        for x in c.lyco_files:
            fi.write(fr"{x[0]},{x[1]}{chr(10)}")
    lylacora_files = c.lora_files + c.lyco_files
    with open(c.save_path+"lylacora_models.txt","wt") as fi:
        for x in lylacora_files:
            f = f_split(x[0])
            fi.write(f"<lora:{f[1]}:0.3:0.3>{chr(10)}")


c.lof = [c.lora_dir+f for f in os.listdir(c.lora_dir[:-1:]) if os.path.isfile(c.lora_dir+f) and f[-(f[::-1].find('.')):].lower() in ['ckpt','pt','safetensors']]
c.lyf = [ c.lyco_dir+f for f in os.listdir(c.lyco_dir[:-1:]) if os.path.isfile(c.lyco_dir+f) and f[-(f[::-1].find('.')):].lower() in ['ckpt','pt','safetensors']]


load_lylocora_files()
chk_lylacora = []
with open(c.save_path+"lylacora_hashes.txt","rt") as fi:
    chk_lylacora = [x.split(chr(44)) for x in fi.readlines()]
list(map(
    lambda x: \
        c.lylacora_files.pop(c.lylacora_files.index([x[0],x[1]])) \
        if [x[0],x[1]] in c.lylacora_files \
        else None,
    [x for x in chk_lylacora]))
with open(c.save_path+"lylacora_hashes.txt","wt") as fi:
    for h in c.lylacora_files:
        fi.write(str(f'{h[0]},{h[1]}{chr(10)}'))


middls=[
    #CIVIT AI MODEL NUMBER LIST HERE
]
err = []
print(len(middls))
_lora_get_ = []
for x in tqdm(np.unique(np.array(middls))):
    if c._hash_check(str(x)) not in c.lylacora_files:
        _lora_get_.append(x)
print(len(_lora_get_))
for mdl in tqdm(middls):
    try:
        c._dl(mdl)
    except Exception as e:
        print(e)
        err.append(e)
