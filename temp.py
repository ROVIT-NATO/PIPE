import urllib.request
import sys

def reporthook(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(total_size)), read_so_far, total_size)
        sys.stderr.write(s)
        if read_so_far >= total_size:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (read_so_far,))

urllib.request.urlretrieve ("http://download1650.mediafire.com/byginqzgf2sg/vrir61dv2ed93ty/FlowNet2_checkpoint.pth.tar", "/home/mahdi/PycharmProjects/PIPE/algos/flow_analysis/FlowNet2_src/pretrained/FlowNet2_checkpoint.pth.tar", reporthook=reporthook)






