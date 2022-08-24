import pytube
from pytube.cli import on_progress
import moviepy.editor as mp
from pydub import AudioSegment
import xmltodict
import requests
from collections import deque
import os
from os.path import join
from models import NMT, STT

class Processor:
    def __init__(self, STT_dir, NMT_dir):
        self.STT = STT(STT_dir)
        self.NMT = NMT(NMT_dir)
    
    def download_and_process(self, code):
        name = code[-11:]
        path = f'www.youtube.com/watch?v={name}'
        yt = pytube.YouTube(path, on_progress_callback=on_progress)
        video_dir = join('static', 'videos')
        save_dir = join(video_dir, f'{name}')
        wav_dir = join(save_dir, 'wav')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            return name
        if not os.path.exists(wav_dir):
            os.makedirs(wav_dir)

        yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first().download(save_dir, f'{name}.mp4')
        clip = mp.VideoFileClip(join(save_dir, f'{name}.mp4'))
        clip.audio.write_audiofile(join(save_dir, f'{name}.wav'))

        # XML 자막 다운로드
        # response = requests.get(yt.captions['ko'].url)
        # xml = response.content
        # with open(join(save_dir, f'{name}.xml'), 'wb') as f:
            # f.write(xml)

        # XML to VTT
        # vtt = xml2vtt(xml.decode('utf-8'))
        # with open(join(save_dir, f'{name}.vtt'), "w", encoding='utf-8') as f:
        #     f.write(vtt)
        
        wav = AudioSegment.from_wav(join(save_dir, f'{name}.wav'))
        durations = make_durations(wav)
        wav_paths = split_wav(wav, durations, wav_dir)
        
        transcriptions = []
        for wav_path in wav_paths:
            transcriptions.append(self.STT.transcribe(wav_path))

        # 5초 단위로 자막이 생성된다. [자막, 시작, 끝]
        parsed_scripts = parse_transcriptions(transcriptions)

        # 생성된 한글 자막 번역
        translated_scripts = []
        for i, script in enumerate(parsed_scripts):
            translated_script = self.NMT.translate(script[0])
            translated_scripts.append([translated_script + '\n' + parsed_scripts[i][0], script[1], script[2]])

        ko_vtt = scripts2vtt(parsed_scripts)
        en_vtt = scripts2vtt(translated_scripts)

        with open(join(save_dir, f'{name}_ko.vtt'), "w", encoding='utf-8') as f:
            f.write(ko_vtt)
        with open(join(save_dir, f'{name}_en.vtt'), "w", encoding='utf-8') as f:
            f.write(en_vtt)

        return name

def ms2time(ms):
    second = ms * 0.001
    minute = second // 60
    second = second % 60
    hour = int(minute // 60)
    minute = int(minute % 60)
    if hour == 0:
        return f'{minute:02d}:{second:06.3f}'
    else:
        return f'{hour:02d}:{minute:02d}:{second:06.3f}'

def xml2vtt(xml):
    cap_dict = xmltodict.parse(xml)
    cap_list = cap_dict['timedtext']['body']['p']
    vtt = 'WEBVTT\n'
    for cap in cap_list:
        start = int(cap['@t'])
        duration = int(cap['@d'])
        end = start + duration
        text = cap['#text']
        vtt += f'\n{ms2time(start)} --> {ms2time(end)}\n'
        vtt += f'{text}\n'
    return vtt

def scripts2vtt(scripts):
    vtt = 'WEBVTT\n'
    for script in scripts:
        text, start, end = script
        vtt += f'\n{ms2time(start)} --> {ms2time(end)}\n'
        vtt += f'{text}\n'
    return vtt

def make_durations(wav, size=60):
    duration = wav.frame_count() / wav.frame_rate
    durations = []
    quotient = int(duration // size)
    remainder = duration % size
    for i in range(quotient):
        durations.append([i * size, (i + 1) * size])
    durations.append([(i + 1) * size, (i + 1) * size + remainder])
    return durations

def split_wav(wav, durations, save_dir):
    paths = []
    for i, (start, end) in enumerate(durations):
        new_wav = wav[int(start*1000):int(end*1000)]
        wav_path = join(save_dir, f'{i:04d}.wav')
        new_wav.export(wav_path, format='wav')
        paths.append(wav_path)
    return paths


# def split_cap_and_wav(wav, cap_list, save_dir):
#     # I need cap lists.
#     cnt = 0
#     for cap in cap_list:
#         # 텍스트 전처리
#         text = cap['#text']
#         # text = re.sub(pattern=pattern, repl='', string=text)
#         text = text.replace('\n', '')
#         text = text.strip()
#         if text == '':
#             continue

#         start = int(cap['@t'])
#         duration = int(cap['@d'])
#         end = start + duration

#         new_name = f'{cnt:06d}.wav'
#         new_path = join(save_dir, new_name)
#         if not os.path.exists(join('wav', save_dir)):
#             os.makedirs(join('wav', save_dir))
#         new_wav = wav[start:end]
#         new_wav.export(new_path, format='wav')
        
#         manifest += f'{new_path}\t{text}\t.\n' 
#         cnt += 1
    
#     return manifest

def parse_transcriptions(transcriptions):
    sentences = []
    for idx, _t in enumerate(transcriptions):
        t = _t[0]
        words = deque()
        word = ''
        start = True
        start_timestamp = 0
        for i, c in enumerate(t['transcription']):
            if c == ' ':
                words.append([word, start_timestamp, t['end_timestamps'][i]])
                word = ''
                start = True
                if i == len(t['transcription'])-1:
                    break
                else:
                    start_timestamp = t['end_timestamps'][i+1]
            else:
                if start:
                    start = False
                word += c
                if i == len(t['transcription'])-1:
                    words.append([word, start_timestamp, t['end_timestamps'][i]])
                    break

        
        for i in range(12):
            end_time = (i + 1) * 5 * 1000
            sentence = ''
            start = True
            start_timestamp = 0
            end_timestamp = 0
            while True:
                if len(words) == 0:
                    break
                word = words.popleft()
                if word[1] < end_time:
                    if start:
                        start = False
                        start_timestamp = word[1] + idx * 60000
                    sentence = sentence + word[0] + ' '
                    end_timestamp = word[2] + idx * 60000
                else:
                    sentences.append([sentence.strip(), start_timestamp, end_timestamp])
                    break
                if len(words) == 0:
                    sentences.append([sentence.strip(), start_timestamp, end_timestamp])
                    break
            
    return sentences

