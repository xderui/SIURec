## SIURec
Implementation of SIURec for paper "Sub-Interest-Aware Representation Uniformity for Recommender System". 

## Run
python main.py --dataset ml-1m

## Efficiency Performance
MovieLens-1M
| Model | Per epoch | Epochs | Total |
| ------ | ----- | ----- | ----- |
AdaGCL  | 15.4544s | 133 | 2055.4305s |
BOD | 2.3784s | 765 | 1819.4985s |
ProtoAU | 4.4132s | 558 | 2462.5736s |
Ours | 3.6809s | 290 | 1067.4749s |

Amazon-Beauty
| Model | Per epoch | Epochs | Total |
| ------ | ----- | ----- | ----- |
AdaGCL | 5.4735  | 217 | 1187.7545s 
BOD | 1.5784  | 402 | 634.5286s
ProtoAU | 3.6826  | 685 | 2522.5769s 
Ours | 2.3478  | 375 | 880.4312s

Amazon-Book
| Model | Per epoch | Epochs | Total |
| ------ | ----- | ----- | ----- |
SimGCL | 4.2848s | 707 | 3029.3204s
XSimGCL | 1.8264s | 851 | 1554.2843s
BOD | 4.3125s | 1392 | 6003.0306s
GraphAU | 2.3213s | 571 | 1325.4338s
Ours | 7.5542s | 376 | 2840.3719s

Gowalla
| Model | Per epoch | Epochs | Total |
| ------ | ----- | ----- | ----- |
SimGCL | 2.8394s | 457 | 1297.5941s 
XSimGCL | 1.2452s | 877 | 1092.0509s 
BOD | 2.4125s | 986 | 2378.7508s 
GraphAU | 2.3421s | 431 | 1009.4559s 
Ours | 4.6052s | 373 | 1717.7580s 