import pandas as pd
import os
import tagme
import sys
import warnings

warnings.filterwarnings('error')

tks = ['5c8abe27-c482-4a9d-8e78-f74baca39b27-843339462', '8ea45e82-c07f-452a-867a-e16fed5be4bb-843339462']

tagme.GCUBE_TOKEN = tks[1]
threshold = 0.1


def compress(inputpath, outputpath):
    dataset = pd.read_json(inputpath, lines=True, encoding='utf-8')[['id', 'title', 'paperAbstract']]
    dataset.to_csv(outputpath, index=False, encoding='utf-8')


def link_one(title: str, abstract: str) -> (str, {str}, {str}):
    try:
        title_annotation = tagme.annotate(title) if title is not '' else None
        title_es = title_annotation.get_annotations(threshold, ) if title_annotation is not None else []
        title_es = {e.entity_title.split('(')[0].strip(' ') for e in title_es}
    except Warning:
        print('warning')
    except Exception as e:
        print(e)
    try:
        abs_annotation = tagme.annotate(abstract) if abstract is not '' else None
        abs_es = abs_annotation.get_annotations(threshold) if abs_annotation is not None else []
        abs_es = {e.entity_title.split('(')[0].strip(' ') for e in abs_es}
    except Warning:
        print('warning')
    except Exception as e:
        print(e)

    return title_es, abs_es


def storeone(res, outroot):
    with open(os.path.join(outroot, res[0]), 'w', encoding='utf-8') as f:
        f.write(res[0] + '`' + ','.join(res[1]) + '`' + ','.join(res[2]) + '\n')


def proc(datapath, resultpath):
    dataset = pd.read_csv(datapath, keep_default_na=False, index_col='id')
    if os.path.isfile(resultpath):
        result = pd.read_csv(resultpath, keep_default_na=False, index_col='id')
    else:
        result = pd.DataFrame.from_dict({'title': [], 'paperAbstract': []})
    count = 0

    mapped_ids = set(result.index)

    try:
        for index, data in dataset.iterrows():
            count += 1
            if count % 100 == 0:
                print(count, flush=True)
                result.to_csv(resultpath, index_label='id', encoding='utf-8')
            if index not in mapped_ids:
                print(count, flush=True)
                try:
                    title_es, abs_es = link_one(data['title'], data['paperAbstract'])
                    result.loc[index] = [','.join(title_es), ','.join(abs_es)]
                except TimeoutError:
                    print('timeout')
    except Exception as e:
        result.to_csv(resultpath, index_label='id', encoding='utf-8')
        print('Save current %d rows' % result.shape[0], flush=True)
        raise e


if __name__ == '__main__':
    data = sys.argv[1]
    csv_path = sys.argv[2]
    result_path = sys.argv[3]

    print('Compress data from %s into %s, tagme result in %s' % (data, csv_path, result_path), flush=True)

    if not os.path.isfile(csv_path):
        compress(data, csv_path)
        print('finish compress', flush=True)

    proc(csv_path, result_path)
