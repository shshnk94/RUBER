import sys
import os
import pandas as pd
import glob
import argparse
import re

class Parser: 

    def __init__(self, datapath, savepath, confpath): 
        
        self.datapath = datapath
        self.savepath = savepath
        self.confpath = confpath
        
        self._read()

    def _read(self): 
        
        self.files = {}
        for dname in glob.glob(os.path.join(self.datapath, '*/')):
            self.files.update({fname.split('_')[-1].split('.')[0]: fname for fname in glob.glob(os.path.join(dname, '*'))})

    def process(self):

        df = pd.read_csv(os.path.join(self.confpath, 'swda.conf'))
        modes = list(df['mode'].unique())

        for mode in modes:

            path = os.path.join(self.savepath, mode) 
            os.makedirs(path, exist_ok=True)

            filenums = list(df.loc[df['mode'] == mode, 'filenum'])
            self._parse(path, filenums)
    
    def _parse(self, path, filenums):
  
        for fnum in filenums:
            
            fname = self.files[str(fnum)]
            df = pd.read_csv(fname)[['caller', 'text']]
            df['text'] = df['text'].apply(self._clean)
            df.to_csv(os.path.join(path, fname.split('/')[-1]), index=False)

    def _clean(self, text): 

        if text.startswith('<') and text.endswith('>.'): 
            return text
        if "[" in text or "]" in text:
            stat = True
        else: 
            stat = False
        group = re.findall("\[.*?\+.*?\]", text)
        while group and stat: 
            for elem in group: 
                elem_src = elem
                elem = re.sub('\+', '', elem.lstrip('[').rstrip(']'))
                text = text.replace(elem_src, elem)
            if "[" in text or "]" in text: 
                stat = True
            else: 
                stat = False
            group = re.findall("\[.*?\+.*?\]", text)
        if "{" in text or "}" in text: 
            stat = True
        else: 
            stat = False
        group = re.findall("{[A-Z].*?}", text)
        while group and stat: 
            child_group = re.findall("{[A-Z]*(.*?)}", text)
            for i in range(len(group)):  
                text = text.replace(group[i], child_group[i])
            if "{" in text or "}" in text: 
                stat = True
            else: 
                stat = False
            group = re.findall("{[A-Z].*?}", text)
        if "(" in text or ")" in text: 
            stat = True
        else: 
            stat = False
        group = re.findall("\(\(.*?\)\)", text)
        while group and stat: 
            for elem in group: 
                if elem: 
                    elem_clean = re.sub("\(|\)", "", elem)
                    text = text.replace(elem, elem_clean)
                else: 
                    text = text.replace(elem, "mumblex")
            if "(" in text or ")" in text:
                stat = True
            else: 
                stat = False
            group = re.findall("\(\((.*?)\)\)", text)

        group = re.findall("\<.*?\>", text)
        if group: 
            for elem in group: 
                text = text.replace(elem, "")

        text = re.sub(r" \'s", "\'s", text)
        text = re.sub(r" n\'t", "n\'t", text)
        text = re.sub(r" \'t", "\'t", text)
        text = re.sub(" +", " ", text)
        text = text.rstrip('\/').strip().strip('-')
        text = re.sub("\[|\]|\+|\>|\<|\{|\}", "", text)
        return text.strip().lower()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Switchboard Dialogue Act Corpus')

    parser.add_argument('--datapath', type=str, help='directory containing data')
    parser.add_argument('--savepath', type=str, help='directory storing data')
    parser.add_argument('--confpath', type=str, help='directory containing the swda.conf file')
    
    args = parser.parse_args()

    model = Parser(args.datapath, args.savepath, args.confpath)
    model.process()
