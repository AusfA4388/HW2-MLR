import json, pathlib
t=pathlib.Path('src/template.html').read_text(encoding='utf-8')
core=pathlib.Path('src/core.js').read_text(encoding='utf-8')
data='const LIFE_CSV = %s;\nconst CAR_CSV = %s;' % (json.dumps(pathlib.Path('data/life.csv').read_text(encoding='utf-8')), json.dumps(pathlib.Path('data/car.csv').read_text(encoding='utf-8')))
out=t.replace('/*__CORE__*/',core).replace('/*__DATA__*/',data)
pathlib.Path('index.html').write_text(out,encoding='utf-8'); print(len(out))
