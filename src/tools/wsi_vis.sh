#!/bin/bash
for i in {1..500}
do
   python src/wsi_pred_visualization.py -w $i
done