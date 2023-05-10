import numpy as np
import os
from imageio import imsave
from argparse import ArgumentParser

def main(args):
	print(args)
	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)
	data = np.load(args.npz_file)
	imgs = data["arr_0"]
	labels = data["arr_1"]
	for i in range(imgs.shape[0]):
		imsave(
			os.path.join(args.out_dir, "c%d_%d.png" % (labels[i], i)),
			imgs[i]
			)


def get_parser():
	ps = ArgumentParser("show images")
	ps.add_argument("--npz-file", type=str, default="/data/marre-diffusion/marre_in64_mw10/s_e20_cs3/samples_50000x64x64x3.npz")
	ps.add_argument("--out-dir", type=str, default="results_e20_cs3")

	return ps

if __name__ == "__main__":
	ps = get_parser()
	args = ps.parse_args()
	main(args)