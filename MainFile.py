from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import statistics
#from pygsr import Pygsr
import numpy

from pylab import plot, show, title, xlabel, ylabel, subplot, savefig
from scipy import fft, arange, ifft
from numpy import sin, linspace, pi
from scipy.io.wavfile import read,write
import numpy as np
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import pyaudio
import wave
#
# (rate,sig) = wav.read("1.wav")
# mfcc_feat = mfcc(sig,rate)
# fbank_feat = logfbank(sig,rate)
# print(mfcc_feat)
# (rate,sig) = wav.read("2.wav")
# mfcc_feat = mfcc(sig,rate)
# fbank_feat = logfbank(sig,rate)
# print(mfcc_feat)
CHUNK = 1024
FORMAT = pyaudio.paInt16 #paInt8
CHANNELS = 2
RATE = 44100 #sample rate
RECORD_SECONDS = 3
WAVE_OUTPUT_BACKGROUND_FILENAME = "background.wav"
WAVE_OUTPUT_AUDIO_FILENAME = "recording.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK) #buffer
print("You must speak from any of these words: 1)monday 2)tuesday 3)wednesday 4) thursday 5) friday 6)saturday 7) sunday")
print("when it prompts to recording speech")
print("* recording background")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data) # 2 bytes(16 bits) per channel

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()
wf = wave.open(WAVE_OUTPUT_BACKGROUND_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK) #buffer

print("* recording speech")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data) # 2 bytes(16 bits) per channel

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_AUDIO_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
# (rate,sig) = wav.read("output.wav")
# mfcc_feat = mfcc(sig,rate)
# fbank_feat = logfbank(sig,rate)
    # #print(mfcc_feat)
    # #print(len(mfcc_feat))
    # #print(len(mfcc_feat[0]))
    # mfcc1=mfcc_feat[0][0]
    # print(mfcc1)
    # for i in mfcc_feat[0]:
    #     print(i)

    # # read audio samples
    # input_data = read("output.wav")
    # audio = input_data[1]
    # # plot the first 1024 samples
    # plt.plot(audio[0:3000])
    # # label the axes
    # plt.ylabel("Amplitude")
    # plt.xlabel("Time")
    # # set the title
    # plt.title("Sample Wav")
    # # display the plot
    # plt.show()
    # speech = Pygsr()
    # speech.record(3) # duration in seconds (3)
    # phrase, complete_response = speech.speech_to_text('es_ES') # select the language
    # print (phrase)

spf = wave.open('recording.wav','r')
    #Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
fs = spf.getframerate()

    #If Stereo
    # if spf.getnchannels() == 2:
    #     print ("Just mono files")
    #     sys.exit(0)
import speech_recognition as sr
r = sr.Recognizer()
with sr.AudioFile("recording.wav") as source:
    print("Recognizing Speech:")
    audio = r.record(source)

try:
    s = r.recognize_google(audio)
    print("Text: "+s)
except Exception as e:
    print("Exception: "+str(e))





def plotSpectru(y,Fs):
 #print(y)
 n = len(y) # lungime semnal
 #print(n)
 k = arange(n)
 #print(k)
 T = n/Fs
 frq = k/T # two sides frequency range
 print(frq)
 frq=frq[30000:120000]
 Y = fft(y)/n # fft computing and normalization

 #print(len(Y))
 Y=abs(Y)
 Y=Y[30000:120000]
 #print(Y)

 # print(max(abs(Y)))
 # print(min(abs(Y)))
 x=sum(Y)
 print(sum(Y))
 print(numpy.mean(abs(Y)))
 plot (frq, abs(Y), 'r') # plotting the spectrum
 xlabel('Freq (Hz)')
 ylabel('|Y(freq)|')
 return x

Fs = 44100;  # sampling rate

rate,data=read('background.wav')
y=data[:,1]
timp=len(y)/44100
t=linspace(0,timp,len(y))

subplot(2,2,1)
plot(t,y)
xlabel('Time')
ylabel('Amplitude')
subplot(2,2,2)
background=plotSpectru(y,Fs)
rate,data=read('recording.wav')
y=data[:,1]
timp=len(y)/44100
t=linspace(0,timp,len(y))

subplot(2,2,3)
plot(t,y)
xlabel('Time')
ylabel('Amplitude')
subplot(2,2,4)
inp=plotSpectru(y,Fs)

print(background,inp)
background=background/1000
inp = inp/1000
d = np.ceil(background/5)
multiplier=2.08-0.12*d
if background*multiplier>inp:
    print("Genuine")
else:
    print("Replay Attack!")


show()


#
# pure = np.linspace(-1, 1, 100)
# plot(pure,arange(100))
# show()
# noise = np.random.normal(10000, 5200, 100)
# plot(arange(100),noise)
# show()
# signal = pure + noise
# plot(arange(100),signal)
# show()