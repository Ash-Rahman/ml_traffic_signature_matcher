import csv
import time

allFlows = []
allCsvHeaders = ['src_ip', 'src_port', 'dst_ip', 'dst_port',
                 'proto', 'pktTotalCount', 'octetTotalCount',
                 'totalFin', 'totalSyn', 'totalRst', 'totalPush',
                 'totalAck', 'totalUrg', 'totalEce', 'totalCwr',
                 'min_ps', 'max_ps', 'avg_ps', 'std_dev_ps',
                 'flowStart', 'flowEnd', 'flowDuration',
                 'min_piat', 'max_piat', 'avg_piat', 'std_dev_piat',

                 'f_pktTotalCount', 'f_octetTotalCount',
                 'f_totalFin', 'f_totalSyn', 'f_totalRst', 'f_totalPush',
                 'f_totalAck', 'f_totalUrg', 'f_totalEce', 'f_totalCwr',
                 'f_min_ps', 'f_max_ps', 'f_avg_ps', 'f_std_dev_ps',
                 'f_flowStart', 'f_flowEnd', 'f_flowDuration',
                 'f_min_piat', 'f_max_piat', 'f_avg_piat', 'f_std_dev_piat',

                 'b_pktTotalCount', 'b_octetTotalCount',
                 'b_totalFin', 'b_totalSyn', 'b_totalRst', 'b_totalPush',
                 'b_totalAck', 'b_totalUrg', 'b_totalEce', 'b_totalCwr',
                 'b_min_ps', 'b_max_ps', 'b_avg_ps', 'b_std_dev_ps',
                 'b_flowStart', 'b_flowEnd', 'b_flowDuration',
                 'b_min_piat', 'b_max_piat', 'b_avg_piat', 'b_std_dev_piat'
                 ]
flow = dict()
flow = {
    'src_ip': 0, 'src_port': 0, 'dst_ip': 0, 'dst_port': 0, 'proto': 0,
    'pktTotalCount': 0, 'octetTotalCount': 0, 'totalFin': 0, 'totalSyn': 0, 'totalRst': 0,
    'totalPush': 0, 'totalAck': 0, 'totalUrg': 0, 'totalEce': 0, 'totalCwr': 0, 'min_ps': 0,
    'max_ps': 0, 'avg_ps': 0, 'std_dev_ps': 0, 'flowStart': 0, 'flowEnd': 0,
    'flowDuration': 0, 'min_piat': 0, 'max_piat': 0, 'avg_piat': 0, 'std_dev_piat': 0,

    'f_pktTotalCount': 0, 'f_octetTotalCount': 0, 'f_totalFin': 0, 'f_totalSyn': 0,
    'f_totalRst': 0, 'f_totalPush': 0, 'f_totalAck': 0, 'f_totalUrg': 0, 'f_totalEce': 0,
    'f_totalCwr': 0, 'f_min_ps': 0, 'f_max_ps': 0, 'f_avg_ps': 0, 'f_std_dev_ps': 0,
    'f_flowStart': 0, 'f_flowEnd': 0, 'f_flowDuration': 0, 'f_min_piat': 0,
    'f_max_piat': 0, 'f_avg_piat': 0, 'f_std_dev_piat': 0,

    'b_pktTotalCount': 0, 'b_octetTotalCount': 0, 'b_totalFin': 0, 'b_totalSyn': 0,
    'b_totalRst': 0, 'b_totalPush': 0, 'b_totalAck': 0, 'b_totalUrg': 0, 'b_totalEce': 0,
    'b_totalCwr': 0, 'b_min_ps': 0, 'b_max_ps': 0, 'b_avg_ps': 0, 'b_std_dev_ps': 0,
    'b_flowStart': 0, 'b_flowEnd': 0, 'b_flowDuration': 0, 'b_min_piat': 0,
    'b_max_piat': 0, 'b_avg_piat': 0, 'b_std_dev_piat': 0,
}


def openCsvFileAndPickFeatures(csvFileName):

    with open(csvFileName) as csvDataFile:
        csvReader = csv.DictReader(csvDataFile)
        for line in csvReader:
            for field in allCsvHeaders:
                flow[field] = line[field]
            allFlows.append(flow.copy())
        return allFlows

# 'src_ip', 'src_port', 'dst_ip', 'dst_port',
# 'proto', 'pktTotalCount', 'octetTotalCount',
# 'totalFin', 'totalSyn', 'totalRst', 'totalPush',
# 'totalAck', 'totalUrg', 'totalEce', 'totalCwr',
# 'min_ps', 'max_ps', 'avg_ps', 'std_dev_ps',
# 'flowStart', 'flowEnd', 'flowDuration',
# 'min_piat', 'max_piat', 'avg_piat', 'std_dev_piat',
#
# 'f_pktTotalCount', 'f_octetTotalCount',
# 'f_totalFin', 'f_totalSyn', 'f_totalRst', 'f_totalPush',
# 'f_totalAck', 'f_totalUrg', 'f_totalEce', 'f_totalCwr',
# 'f_min_ps', 'f_max_ps', 'f_avg_ps', 'f_std_dev_ps',
# 'f_flowStart', 'f_flowEnd', 'f_flowDuration',
# 'f_min_piat', 'f_max_piat', 'f_avg_piat', 'f_std_dev_piat',
#
# 'b_pktTotalCount', 'b_octetTotalCount',
# 'b_totalFin', 'b_totalSyn', 'b_totalRst', 'b_totalPush',
# 'b_totalAck', 'b_totalUrg', 'b_totalEce', 'b_totalCwr',
# 'b_min_ps', 'b_max_ps', 'b_avg_ps', 'b_std_dev_ps',
# 'b_flowStart', 'b_flowEnd', 'b_flowDuration',
# 'b_min_piat', 'b_max_piat', 'b_avg_piat', 'b_std_dev_piat',
