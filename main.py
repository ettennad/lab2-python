import wave
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

# ====== Чтение файла ======
wav = wave.open("8.wav", 'r')
framerate = wav.getframerate() #читает частоту дискретизации
n_frames = wav.getnframes() #кол-во отсчетов
data = np.frombuffer(wav.readframes(n_frames), dtype=np.int16) #массив амплитуд
wav.close()

# ====== Ввод ======
n = int(input("Количество отсчетов на круговой диаграмме:"))

# ====== 1.1 Круговая диаграмма ======
samples = data[:n]
amps = np.abs(samples).astype(float)

total = amps.sum()
labels = []
for i in range(n):
    if amps[i] / total > 0.015:
        labels.append(f'{i}({samples[i]})')
    else:
        labels.append('')

plt.figure(figsize=(14, 12))
plt.pie(amps, 
        labels=labels,
        startangle=90,
        counterclock=False,
        labeldistance=0.65,
        rotatelabels=True,
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.5},
        textprops={'fontsize': 6})
plt.title(f'Круговая диаграмма (количество отсчетов {n})\n'
          'Площадь сектора пропорциональна амплитуде',
          fontsize=14)
plt.show()

# ====== 1.2 Осциллограмма ======
t = np.linspace(0, len(data)/framerate, len(data))
plt.figure(figsize=(12, 4))
plt.plot(t, data, 'b-', linewidth=0.5)
plt.title('Осциллограмма речевого сигнала')
plt.xlabel('Время, с')
plt.ylabel('Амплитуда (16-битные отсчеты)')
plt.grid(True, alpha=0.3)
plt.show()

# ====== 1.3 Спектр ln(Re² + Im²) ======
dft = np.fft.fft(data)
freqs = np.fft.fftfreq(len(data), 1/framerate)
mask = freqs >= 0
freqs = freqs[mask] #маска из библиотеки numpy (при true элемент записывается)
dft = dft[mask]
log_power = np.log(dft.real**2 + dft.imag**2 + 1e-10)

plt.figure(figsize=(12, 5))
plt.plot(freqs, log_power, 'r-', linewidth=0.8)
plt.title('Спектр: логарифм квадрата модуля ДПФ\nln(Re² + Im²)')
plt.xlabel('Частота, Гц')
plt.ylabel('ln(Re² + Im²)')
plt.grid(True, alpha=0.3)
plt.show()

# ====== 1.4 Гистограмма ======
plt.figure(figsize=(10, 5))
plt.hist(data, bins=100, color='green', alpha=0.7, edgecolor='black')
plt.title('Гистограмма распределения амплитуд отсчетов')
plt.xlabel('Амплитуда')
plt.ylabel('Количество отсчетов в интервале')
plt.grid(True, alpha=0.3, axis='y')
plt.show()

print(f"\nВремя выполнения: {time.time() - start_time:.2f} сек")

#процент ии - 100%