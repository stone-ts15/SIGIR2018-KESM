
import scrapy
import json
from spider.items import *
import os
import pandas as pd


class ScholarSpider(scrapy.Spider):
    name = 'scholar'
    allowed_domains = ['*']
    start_urls = [
        'https://www.wikidata.org/w/api.php'
    ]

    def __init__(self, category=None, *args, **kwargs):
        super(ScholarSpider, self).__init__(*args, **kwargs)
        # properties
        self.url_format = 'https://www.wikidata.org/w/api.php?action=wbsearchentities&search=%s&language=en&limit=1&format=json'
        self.data_path = '/root/dataset/art/div10000-aa'
        self.abac_path = '/root/dataset/art/tagme-abac.csv'

        self.mapped_entities = set()

        if os.path.isfile('maps.txt'):
            with open('maps.txt', 'r', encoding='utf-8') as m:
                for line in m.readlines():
                    self.mapped_entities.add(line.split('`')[0])

        self.entities = self.mapped_entities.copy()

        # with open(self.data_path, 'r', encoding='utf-8') as f:
        #     for line in f.readlines():
        #         _, title, abst = line[:-1].split('`')
        #         self.entities.update(title.split(','))
        #         self.entities.update(abst.split(','))

        # turn to read ab/ac files, maps.txt already contains all entities in div-aa

        abac = pd.read_csv(self.abac_path, keep_default_na=False, index_col='id')
        for index, data in abac.iterrows():
            self.entities.update(data['title'].split(','))
            self.entities.update(data['paperAbstract'].split(','))

        self.entities.difference_update(self.mapped_entities)
        self.entities.remove('')
        print('Target entities num: %d' % len(self.entities), flush=True)

    def start_requests(self):
        header = {'Content-Type': 'application/json'}
        for e in self.entities:
            url = self.url_format % e
            yield scrapy.Request(url, meta={'search_key': e}, headers=header, callback=self.parse_search)

    def parse_search(self, response: scrapy.http.Response):
        # example:
        # https://www.wikidata.org/w/api.php?action=wbsearchentities&search=Obama&language=en&limit=1&format=json
        item = ScholarItem()
        search = json.loads(response.body.decode('utf-8'))
        item['entity'] = response.meta['search_key']
        try:
            item['description'] = search['search'][0]['description']
        except:
            item['description'] = ''
        yield item
