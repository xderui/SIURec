## SIURec
Implementation of SIURec for paper "Sub-Interest-Aware Representation Uniformity for Recommender System". 

## Run
python main.py --dataset ml-1m

## Efficiency Performance (with Recall and NDCG results)
MovieLens-1M
| Model | Per epoch | Epochs | Total | Recall@20 | NDCG@20 | 
| ------ | ----- | ----- | ----- | ----- | ----- |
AdaGCL  | 15.4544s | 133 | 2055.4305s | 0.1035 | 0.0811 |
BOD | 2.3784s | 765 | 1819.4985s | 0.0963 | 0.0798 | 
ProtoAU | 4.4132s | 558 | 2462.5736s | 0.1003 | 0.0811 |
Ours | 3.6809s | 290 | 1067.4749s | 0.1099 | 0.0877 | 

Amazon-Beauty
| Model | Per epoch | Epochs | Total | Recall@20 | NDCG@20 | 
| ------ | ----- | ----- | ----- | ----- | ----- |
AdaGCL | 5.4735  | 217 | 1187.7545s |  0.1268 | 0.0690 |
BOD | 1.5784  | 402 | 634.5286s | 0.1348 | 0.0752 |
ProtoAU | 3.6826  | 685 | 2522.5769s | 0.1193 | 0.0651 |
Ours | 2.3478  | 375 | 880.4312s | 0.1374 | 0.0759 |

Amazon-Book
| Model | Per epoch | Epochs | Total | Recall@20 | NDCG@20 | 
| ------ | ----- | ----- | ----- | ----- | ----- |
SimGCL | 4.2848s | 707 | 3029.3204s | 0.0892 | 0.0682 |
XSimGCL | 1.8264s | 851 | 1554.2843s | 0.0893 | 0.0676 |
BOD | 4.3125s | 1392 | 6003.0306s | 0.0847 | 0.0647 |
GraphAU | 2.3213s | 571 | 1325.4338s | 0.0849 | 0.0648 |
Ours | 7.5542s | 376 | 2840.3719s | 0.0944 | 0.0722 |

Gowalla
| Model | Per epoch | Epochs | Total | Recall@20 | NDCG@20 | 
| ------ | ----- | ----- | ----- | ----- | ----- |
SimGCL | 2.8394s | 457 | 1297.5941s | 0.1930 | 0.1156 |
XSimGCL | 1.2452s | 877 | 1092.0509s | 0.1956 | 0.1165 |
BOD | 2.4125s | 986 | 2378.7508s | 0.1913 | 0.1143 |
GraphAU | 2.3421s | 431 | 1009.4559s | 0.1821 | 0.1070 |
Ours | 4.6052s | 373 | 1717.7580s | 0.2048 | 0.1224 | 