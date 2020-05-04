#!/usr/bin/env bash
# export src_dir='/data/vectorization/datasets/svg_datasets/whole_images/abc/test'
# export dst_dir='/data/field_learn/datasets/svg_datasets/patched/abc/simple/test'
export src_dir=$1
export dst_dir=$2

make_patches(){
	src="$1"
	filename="$(basename "$src")"
	echo "Processing $filename"
	python make_patches.py -i "$src" -o "$dst_dir" --patch-height=128 --patch-width=128 --image-scale=2 --num-augmentations=4
}
export -f make_patches

find $src_dir -type f -name '*.svg' | /run/user/bin/bin/parallel make_patches
