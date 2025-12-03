

import numpy as np

from scapy.compat import raw
from scapy.layers.inet import IP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Padding
import os

from utils import should_omit_packet, read_pcap, PREFIX_TO_APP_ID, PREFIX_TO_TRAFFIC_ID
from utils import PREFIX_TO_TorApp_ID, ID_TO_APP, ID_TO_TRAFFIC
from tqdm import tqdm
import time 
import random




def remove_ether_header(packet):
    if Ether in packet:
        return packet[Ether].payload

    return packet


def mask_ip(packet):
    if IP in packet:
        packet[IP].src = '0.0.0.0'
        packet[IP].dst = '0.0.0.0'

    return packet


def pad_udp(packet):
    if UDP in packet:
        # get layers after udp
        layer_after = packet[UDP].payload.copy()

        # build a padding layer
        pad = Padding()
        pad.load = '\x00' * 12

        layer_before = packet.copy()
        layer_before[UDP].remove_payload()
        packet = layer_before / pad / layer_after

        return packet

    return packet

def packet_to_sparse_array(packet, max_length=1480):
    arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length] #/ 255
    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)
    return arr


def transform_packet(packet):

    packet = remove_ether_header(packet)
    packet = pad_udp(packet)
    packet = mask_ip(packet)
    
    arr = packet_to_sparse_array(packet)
    
    if arr is not None:
        token = ""
        for i in arr:
            token = token + " " + str(i) 
        return token.strip(" ")

def transform_pcap(path):
    # Service 11 分类
    # App 12 分类
    f_service = open("service/datanet_service.txt",'a') #文件存储路径
    f_app = open("app/datanet_app.txt",'a')

    c = 0
    prefix = path.split('/')[-1].split('.')[0].lower()
    # app_label = PREFIX_TO_APP_ID.get(prefix)       ISCX-VPN 数据集
    # service_label = PREFIX_TO_TRAFFIC_ID.get(prefix)    ISCX-VPN 数据集
    app_label = PREFIX_TO_TorApp_ID.get(prefix)  # tor数据集
    service_label = None   # tor 数据集
    c = 0
    for i, packet in enumerate(read_pcap(path)):
        token = transform_packet(packet)
        
        if token is not None: 
            c += 1
            if app_label is not None :
                f_app.write(token+"\t"+str(app_label)+"\n")
                
                

            if service_label is not None:
                f_service.write(token+"\t"+str(service_label)+"\n")
                      
        if i > 1000 :
            return



def all_path(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
    return result

if __name__ == '__main__':
    source =  "/home/user/xtm/datasets/tor/"  #选择自己的路径  tor的
    # source =  "/home/user/xtm/dataset/raw_pcap/"  # ISCX-VPN 的
    
    root = os.listdir(source)
    random.shuffle(root)
    summ = 0 
    for i in tqdm(root):
   
        path = source + i
        transform_pcap(path)

    