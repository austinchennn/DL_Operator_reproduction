# 22.卷积操作
# 在图像任务中，卷积层要对多通道输入做滑窗加权求和。现在请你实现“多通道、单输出通道”的二维卷积:同一位置上，各输入通道卷积结果累加得到输出值，不做激活与归一化。
# 输入张量形状:(C,H_in,W_in)，共C个通道，每个通道是H_in xW_in的二维数组。
# 卷积核形状:(C,K h,K_w)，每个通道配一张K hxK_w的核，通道数与输入相同。
# 步长与填充:步长为stride(整数>1)，四周以0填padding层。
# 计算方式:
# 1)先在输入四周补 0，得到尺寸(C,H_in+2*padding,W_in+2*padding)。
# 2)以步长stride滑动大小为K h xK_w的窗口;若窗口越界(不足以覆盖核)，该位置跳过。
# 3)对每个窗口:逐通道与对应核做逐元素相乘并求和，再把各通道和相加，得到该格的输出值。
# 输出张量形状为(H out,W out)，其中
# H out=(H in+2*padding-K h)//stride+1
# W_out=(W in+2*padding-K w)//stride +1
# 所有输入与输出均为整数。
import io


def reader_convol(input_string: str):
    """
    读取input与kernel
    :param input:
    :return:
    """
    input = []
    kernel = []
    with io.StringIO(input_string) as f:
        #读input
        input_size = [int(x) for x in f.readline().strip().split()]
        for _ in range(input_size[0]):
            input_channel = []
            for _ in range(input_size[1]):
                row = [int(x) for x in f.readline().strip().split()]
                input_channel.append(row)
            input.append(input_channel)
        # 读kernel
        kernel_size = [int(x) for x in f.readline().strip().split()]
        for _ in range(kernel_size[0]):
            kernel_channel = []
            for _ in range(kernel_size[1]):
                row = [int(x) for x in f.readline().strip().split()]
                kernel_channel.append(row)
            kernel.append(kernel_channel)
        #读data
        other_data = [int(x) for x in f.readline().strip().split()]
        stride = other_data[0]
        padding = other_data[1]
    return [input, kernel,stride, padding]

def padding(input: list, padding: int):
    H = len(input)
    W = len(input[0])
    input_padded = []
    for channel in input:
        curr_padded = []
        #上
        for _ in range(padding):
            curr_padded.append([0]*(W + 2* padding))
        #左右
        for row in channel:
            curr_row_padded = [0] * padding + row + [0] * padding
            curr_padded.append(curr_row_padded)
        # 下
        for _ in range(padding):
            curr_padded.append([0]*(W + 2* padding))
        input_padded.append(curr_padded)
    return input_padded

def single_chanel_convol(input_padded, kernel, stride: int):

    H_in_pad  = len(input_padded)
    W_in_pad = len(input_padded[0])
    K_h = len(kernel)
    K_w = len(kernel[0])

    OH = (H_in_pad - K_h) // stride + 1
    OW = (W_in_pad - K_w) // stride + 1
    output = [[0 for _ in range(OW)] for _ in range(OH)]
    for i in range(OH):
        for j in range(OW):
            h_start = i * stride
            w_start = j * stride
            cur_sum = 0
            for cur_h in range(K_h):
                for cur_w in range(K_w):
                    cur_sum += input_padded[h_start+cur_h][w_start+cur_w] * kernel[cur_h][cur_w]
            output[i][j] = cur_sum
    return output

def multi_channel_convol(input_padded, kernel,stride):
    output = []
    for i in range(len(input_padded)):
        input_channel = input_padded[i]
        kernel_channel = kernel[i]
        output.append(single_chanel_convol(input_channel,kernel_channel,stride))
    return output

def main(input_str:str):
    data_cleaned = reader_convol(input_str)
    input_cleaned = data_cleaned[0]
    kernel_cleaned = data_cleaned[1]
    stride = data_cleaned[2]
    padd = data_cleaned[3]
    #padding
    input_padded = padding(input_cleaned, padd)
    #Convolution:
    output = multi_channel_convol(input_padded,kernel_cleaned,stride)
    for channel in output:
        for line in channel:
            print(" ".join([str(x) for x in line]))




if __name__ == '__main__':
    # print("Process Data:")
    input_str = """ 1 4 4
                    1 0 1 0
                    0 1 0 1
                    1 0 1 0
                    0 1 0 1
                    1 3 3
                    0 1 0
                    1 1 1
                    0 1 0
                    1 1"""
    data_cleaned = reader_convol(input_str)
    input_cleaned = data_cleaned[0]
    kernel_cleaned = data_cleaned[1]
    stride = data_cleaned[2]
    padd = data_cleaned[3]


    print("input:", input_cleaned)
    print("Kernel:", kernel_cleaned)
    print("stride:", stride)
    print("padding:", padd)
    print("______________________________________")
    print("Padding:")
    input_padded = padding(input_cleaned,padd)
    print("input with padding:", input_padded)
    print("______________________________________")
    print("Convolution:")
    output = multi_channel_convol(input_padded,kernel_cleaned,stride)
    print("convoluted output:", output)
    print("______________________________________")
    main(input_str)