#!/usr/bin/env python3
import json, os

def build_classname(index, name, duplicates):
    if name in duplicates:
        return f'{name}-{index}'
    else:
        return name

def build_fullname(index, name):
    return f'{index}-{name}'

def replace_content_with_labels(dictionary, content, replace):
    for k in dictionary.keys():
        classindex = f'c{k.zfill(3)}'
        label = replace(classindex, dictionary[k][1])
        content = content.replace(classindex, label)
        
    return content

def replace_with_labels(dictionary, duplicates, model_file):
    with open(model_file, 'rt') as fin:
        input = fin.read()

    def save(suffix, content):
        output_file = model_file.replace('model.bif', suffix)
        with open(output_file, 'wt') as fout:
            fout.write(content)
        print(f'{output_file} saved.')
        
    model_classname = replace_content_with_labels(
        dictionary, input, lambda i, n: build_classname(i, n, duplicates))
    save('model-classname.bif', model_classname)

    model_fullname = replace_content_with_labels(dictionary, input, build_fullname)
    save('model-fullname.bif', model_fullname)    

    
def change_files(model_files):
    with open('imagenet_class_index.json') as f:
        imagenet_class_index = json.load(f)

    classes = sorted(p[1] for p in imagenet_class_index.values())
    duplicates = [classes[i] for i in range(len(classes) - 1) if classes[i] == classes[i+1]]
    
    for f in model_files:
        replace_with_labels(imagenet_class_index, duplicates, f)    

        
def change_files_in_dir(directory):
    change_files([f for f in os.listdir() if f.endswith('model.bif')])

    
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('use_classnames file [...]')
    else:
        change_files(sys.argv[1:])
