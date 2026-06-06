import json
from pathlib import Path
path = Path(r'G:\Meu Drive\Projetos\Github\cases\case_datarisk\notebooks\01_population.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))
for idx, cell in enumerate(nb['cells'], 1):
    if cell['cell_type'] == 'code' and ('build_active_population' in ''.join(cell['source']) or 'build_population_score' in ''.join(cell['source']) or 'population_active' in ''.join(cell['source']) or 'population_score' in ''.join(cell['source'])):
        print('CELL', idx)
        print(''.join(cell['source']))
        print('---')
