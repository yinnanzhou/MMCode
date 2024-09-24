import numpy as np
import matplotlib.pyplot as plt


def plotMain(data_1dfft, data_1dfft_dyn, range_angle, angle_diff,
             angle_diff_filtered, t_f_Zxx):
    numFrames, numSamples = data_1dfft.shape
    t, f, Zxx = t_f_Zxx
    X, Y = np.meshgrid(np.linspace(1, numSamples, numSamples),
                       np.linspace(1, numFrames, numFrames))

    # # 1D FFT
    # ax = plt.figure(1).add_subplot(projection='3d')
    # ax.plot_surface(X, Y, np.abs(data_1dfft), cmap='viridis')

    # # 去除静态后的 1D FFT
    # ax = plt.figure(2).add_subplot(projection='3d')
    # ax.plot_surface(X, Y, np.abs(data_1dfft_dyn), cmap='viridis')

    # # 类似Matlab的imagesc
    # plt.imshow(abs(data_1dfft_dyn), aspect='auto', cmap='jet', interpolation='none')

    plt.figure(3)
    plt.subplot(4, 1, 1)
    plt.plot(range_angle)
    plt.title('Unwrapped Phase')
    plt.xlabel('Sample Index')
    plt.ylabel('Phase')
    plt.xlim(0, len(range_angle))
    for i in np.linspace(len(range_angle) / 10, len(range_angle), 10):
        plt.axvline(x=i, color='r', linestyle='--')

    plt.subplot(4, 1, 2)
    plt.plot(angle_diff)
    plt.title('Phase Difference')
    plt.xlabel('Sample Index')
    plt.ylabel('Phase Difference')
    plt.xlim(0, len(range_angle))
    for i in np.linspace(len(range_angle) / 10, len(range_angle), 10):
        plt.axvline(x=i, color='r', linestyle='--')

    # 绘制带通滤波前后的相位差
    plt.subplot(4, 1, 3)
    plt.plot(angle_diff_filtered)
    plt.title('Phase Difference - Original and Filtered')
    plt.xlabel('Sample Index')
    plt.ylabel('Filtered Phase Difference')
    plt.xlim(0, len(range_angle))
    for i in np.linspace(len(range_angle) / 10, len(range_angle), 10):
        plt.axvline(x=i, color='r', linestyle='--')

    # 绘制STFT结果
    # plt.figure(4)
    plt.subplot(4, 1, 4)
    plt.imshow(10 * np.log10(np.abs(Zxx)),
               aspect='auto',
               extent=[t.min(), t.max(), f.min(),
                       f.max()],
               origin='lower',
               cmap='jet',
               interpolation='bilinear')
    plt.title('STFT of Filtered Phase Difference')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    # plt.colorbar(label='Magnitude')
    for i in np.linspace(t[(len(t) - 1) // 10], t[-1], 10):
        plt.axvline(x=i, color='r', linestyle='--')
    # plt.ylim([0,np.log10(10000)])
    # plt.tight_layout()

    plt.show()


def plotSTFT(t, f, Zxx):
    plt.figure(figsize=(8, 6))
    plt.imshow(
        10 * np.log10(np.abs(Zxx)),
        aspect='auto',
        extent=[t.min(), t.max(), f.min(), f.max()],
        origin='lower',
        cmap='jet',
        interpolation='bilinear',
        vmin=-60,
        vmax=-15,
    )
    # bilinear lanczos bicubic hamming

    plt.colorbar(label='Power (dB)')
    plt.title('STFT of Filtered Phase Difference')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.show()


def saveSTFT(t, f, Zxx, save_path, wordIndex, fileIndex, personIndex, txIndex):
    num_subplots = 20
    t_intervals = np.linspace(t.min(), t.max(), num_subplots + 1)

    for i in range(num_subplots):
        plt.figure()
        start_time = t_intervals[i]
        end_time = t_intervals[i + 1]
        indices = (t >= start_time) & (t < end_time)

        plt.imshow(
            10 * np.log10(np.abs(Zxx[:, indices])),
            aspect='auto',
            extent=[start_time, end_time,
                    f.min(), f.max()],
            origin='lower',
            cmap='jet',
            interpolation='bilinear',
            vmin=-60,
            vmax=-15,
        )
        # plt.colorbar()
        plt.axis('off')  # Remove axes
        plt.savefig(
            f"{save_path}/{str(wordIndex)}_{str(fileIndex*num_subplots+i)}_{str(personIndex)}_{str(txIndex)}.png",
            bbox_inches='tight',
            pad_inches=0,
            dpi=300)
        plt.close()

