from pathlib import Path
import os

root = '.'
p = Path(root)
thumbs = list(p.rglob('*.db'))
if thumbs:
    for item in thumbs:
        os.remove(item)
        print('{} has been removed'.format(item))
else:
    print('no thumbs.db exists')
