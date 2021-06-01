import argparse
import subprocess

import pcapkit as pcapDump
import pyshark as shark
import os

# from pcap_processor.map_manager import MapperManager
# from pcap_processor.sink import SinkManager

import pcap_processor
from pcap_processor import plugin_manager, commons
from pcap_processor.__main__ import file_type
from pcap_processor.plugin import Plugin
from pcap_processor.reader import PcapReader


def dumpPCAPToJSON(PCAPInputFile, JSONOutputFile):
    print("hello")
    pcapDump.extract(fin=PCAPInputFile, fout=JSONOutputFile, format='json', extension=False)

def dumpPCAPToCSV(PCAPInputFile):

    commons.setup_logging()
    plugin_manager.load_plugins()
    parser = argparse.ArgumentParser(prog="pcap-processor",
                                     description="Read and process pcap files using this nifty tool.")
    plugin_manager.fill_cmd_args(parser)
    parser.add_argument("--version", action="version",
                        version="%(prog)s 0.0.1")
    parser.add_argument("file", type=file_type, nargs="+", help="pcap file to read")

    args = parser.parse_args()
    args.file = PCAPInputFile
    pcap_file = args.file

    plugin_manager.process_config(args)
    reader = PcapReader(pcap_file)
    reader.read()

def readPCAPFile(PCAPInputFile):
    capture = shark.FileCapture(PCAPInputFile)
    for i in capture:
        print(i)
    capture.close()

# def _read_pcap(self, path):
#     # logger.debug("Reading pcap file: %s", path)
#     packets = pyshark.FileCapture(path)
#     for pcap in packets:
#         has_transport = pcap.transport_layer is not None
#         packet_time = float(pcap.sniff_timestamp)
#         packet_dict = dict()
#         highest_layer = pcap.highest_layer.upper()
#         packet_dict["highest_layer"] = highest_layer
#         if has_transport:
#             packet_dict["transport_layer"] = pcap.transport_layer.upper()
#         else:
#             packet_dict["transport_layer"] = "NONE"
#             packet_dict["src_port"] = -1
#             packet_dict["dst_port"] = -1
#             packet_dict["transport_flag"] = -1
#
#         packet_dict["timestamp"] = int(packet_time * 1000)
#         packet_dict["time"] = str(pcap.sniff_time)
#         packet_dict["packet_length"] = int(pcap.length)
#         packet_dict["data"] = ""
#
#         for layer in pcap.layers:
#             layer_name = layer.layer_name.upper()
#             if "IP" == layer_name or "IPV6" == layer_name:
#                 packet_dict["src_ip"] = str(layer.src)
#                 packet_dict["dst_ip"] = str(layer.dst)
#                 if hasattr(layer, "flags"):
#                     packet_dict["ip_flag"] = int(layer.flags, 16)
#                 else:
#                     packet_dict["ip_flag"] = -1
#                 if hasattr(layer, "geocountry"):
#                     packet_dict["geo_country"] = str(layer.geocountry)
#                 else:
#                     packet_dict["geo_country"] = "Unknown"
#
#             elif has_transport and layer_name == pcap.transport_layer:
#                 packet_dict["src_port"] = int(layer.srcport)
#                 packet_dict["dst_port"] = int(layer.dstport)
#                 if hasattr(layer, "flags"):
#                     packet_dict["transport_flag"] = int(layer.flags, 16)
#                 else:
#                     packet_dict["transport_flag"] = -1
#
#             elif "FTP" == layer_name:
#                 packet_dict["data"] = str(layer._all_fields)
#         if "src_ip" not in packet_dict:
#             continue
#         # Map packet attributes
#         packet_dict = MapperManager.map(packet_dict)
#         SinkManager.write(packet_dict)

