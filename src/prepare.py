#!/usr/bin/python3
# -*-coding:utf-8-*-
import os
import json


class CustomDataGenerator(object):
    def __init__(self, root_dir, dump_txt_path):
        self._root_dir = root_dir
        self._dump_txt_path = dump_txt_path

    def run(self):
        annotation_files = [f for f in os.listdir(self._root_dir) if f.endswith('.json') and 'status' not in f]
        samples = list()

        for f_name in annotation_files:
            labels_info = json.load(open(os.path.join(self._root_dir, f_name)))
            for item in labels_info:
                video_name = item['video']
                label = item['label']

                video_full_path = os.path.abspath(os.path.join(
                    self._root_dir,
                    video_name.split('\\')[0],
                    video_name.split('\\')[1]))

                if not os.path.exists(video_full_path):
                    continue
                samples.append(dict(
                    video_full_path=video_full_path,
                    label=label
                ))

        self.dump_train_txt(samples)

    def dump_train_txt(self, samples):
        labels = list(set([s['label'] for s in samples]))
        mapping = dict()
        for i, label in enumerate(labels):
            mapping[label] = i

        with open(self._dump_txt_path, 'w') as wf:
            for sample in samples:
                wf.write('{} {} {}\n'.format(
                    sample['video_full_path'],
                    100,    # any value could be fine, video loader will compute it during training phase
                    mapping[sample['label']]
                ))

        json.dump(mapping, open('../mapping.json', 'w'), ensure_ascii=False, indent=True)


if __name__ == '__main__':
    generator = CustomDataGenerator(
        root_dir='../dataset/',
        dump_txt_path='../train.txt'
    )

    generator.run()



