Generate文件夹 用于读取原始bin数据，转换成频谱图输出

1tx4rx.py 用于读取1个发射端、4个接收端的数据，本数据集不包含这种形式的数据
3tx4rx.py 用于读取3个发射端、4个接收端的数据，生成12张对应的频谱图
3tx4rx_3dfft.py 用于读取3个发射端、4个接收端的数据，利用angleFFT，1个tx、4个rx为一组，一共3组，生成3张对应的频谱图。、

mmgenerate_3tx4rx.ipynb 对所有bin文件进行读取，保存结果，每个bin文件得到12张频谱图
mmgenerate_3tx4rx_3dfft.ipynb 对所有bin文件进行读取，保存结果，每个bin文件得到3张频谱图

图片命名格式:
a_b_c_d.png
    其中a是音标的序号，从0到47
    b是第几条数据，从0到39
    c是说话人的序号，目前只有0
    d是相同数据频谱图的序号，对于processed1D，范围为0到11，对于processed3D，范围为0-2