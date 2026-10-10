"""Install per-user launcher/icon/.vtw MIME association; no system-wide changes."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from app.identity import APP_NAME, DESKTOP_ID, ICON_FILE


def exec_quote(value):
    # Desktop Entry quoting, not shell escaping. Percent is a literal %% field escape.
    quoted=str(value).replace('\\','\\\\').replace('"','\\"').replace('`','\\`').replace('$','\\$').replace('%','%%')
    return '"'+quoted.replace('\\','\\\\')+'"'


def artifacts(data_home, app_root, python):
    desktop = '\n'.join([
        '[Desktop Entry]', 'Type=Application', 'Version=1.0', 'Name='+APP_NAME,
        'Comment=Visual PyTorch architectures and experiments',
        'Exec='+exec_quote(python)+' '+exec_quote(app_root/'run.py')+' %f',
        'Icon='+DESKTOP_ID, 'Terminal=false', 'Categories=Development;Science;',
        'MimeType=application/x-visual-torch-workspace;', 'StartupNotify=true',
        'StartupWMClass='+APP_NAME, 'Keywords=Visual;PyTorch;neural;network;architecture;', ''])
    mime = '''<?xml version="1.0" encoding="UTF-8"?>
<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
  <mime-type type="application/x-visual-torch-workspace">
    <comment>Visual Torch Workspace</comment>
    <sub-class-of type="application/vnd.sqlite3"/>
    <glob pattern="*.vtw" weight="80"/>
    <icon name="tensor-loom"/>
  </mime-type>
</mime-info>
'''
    return {
        data_home/('applications/'+DESKTOP_ID+'.desktop'): desktop,
        data_home/('mime/packages/'+DESKTOP_ID+'.xml'): mime,
        data_home/('icons/hicolor/scalable/apps/'+DESKTOP_ID+'.svg'): (app_root/ICON_FILE).read_text(),
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true',help='Print exact targets without writing')
    parser.add_argument('--data-home',type=Path,default=Path(os.environ.get('XDG_DATA_HOME',Path.home()/'.local/share')))
    args=parser.parse_args(argv)
    args.data_home=args.data_home.expanduser().resolve()
    files=artifacts(args.data_home,Path(__file__).resolve().parent,Path(sys.executable).resolve())
    for path,content in files.items():
        if args.dry_run:
            print(str(path)+'\n'+content)
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(content)
            print('Installed '+str(path))
    if not args.dry_run:
        for command,target in [('update-desktop-database','applications'),('update-mime-database','mime'),('gtk-update-icon-cache','icons/hicolor')]:
            executable=shutil.which(command)
            if executable:
                result=subprocess.run([executable,*(['-f','-t'] if command=='gtk-update-icon-cache' else []),str(args.data_home/target)],capture_output=True,text=True)
                print(command+': '+('OK' if result.returncode==0 else result.stderr.strip()))
            else:
                print('Cache refresh unavailable: '+command)
        print(APP_NAME+' is registered as an Open With candidate for .vtw, not forced as the default.')


if __name__=='__main__':
    try:
        main()
    except OSError as error:
        raise SystemExit('Desktop registration could not complete: '+str(error))
