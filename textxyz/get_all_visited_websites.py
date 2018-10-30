import psycopg2
import re
from bs4 import BeautifulSoup


def connect_and_get_cursor (dbname, user, host, password):

    connect_str = ( ' dbname='   + str(dbname)
                  + ' user='     + str(user)
                  + ' host='     + str(host)
                  + ' password=' + str(password) )
    conn = psycopg2.connect(connect_str)
    cursor = conn.cursor()
    return cursor

def grab(soup):
    text = ''

    for item in soup.findAll(text=True):
            text += item + ' '
    text.replace('   ', '\n')
    return text

def get_all_ids_and_webpages (cursor):
    cursor.execute('SELECT * from this_table_here OFFSET 10000 LIMIT 24000;')
    rows = cursor.fetchall()

    ids = []
    webpages = []
    for row in rows:
        ids.append (row[0])
        webpages.append (row [3])

    return ids, webpages

if __name__ == '__main__':
    cursor = connect_and_get_cursor (dbname = 'testdb',
                                     user   = 'test',
                                     host   = 'www.test.test',
                                     password = 'xywz')

    ids, webpages = get_all_ids_and_webpages (cursor)
    for index in range (len(ids)):
        id = ids [index]
        page = webpages [index]
        soup = BeautifulSoup (page, 'html.parser', from_encoding='utf-8')
        for script in soup (["script", "style"]):
            script.extract ()
        text = grab(soup)
        lines = (line.strip() for line in text.splitlines ())
        chunks = (line.strip() for line in lines)
        text = '\n'.join(chunk for chunk in chunks if chunk)
        filename = './website_data/' + str(id) + '.website'
        file = open (filename, 'w')
        file.write (text)

