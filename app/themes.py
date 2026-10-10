"""Semantic theme tokens shared by native Qt and the embedded graph editor."""
THEME_NAMES = tuple(sorted(('light', 'dark', 'system', 'monokai', 'obsidian', 'github', 'neon', 'oceanic', 'terminal', 'matrix', 'math',
               'retrofuturism','cyberpunk','blueprint','aurora','bauhaus','metallic')))
SERIF_FAMILIES = ('STIX Two Text', 'STIX', 'Noto Serif', 'DejaVu Serif', 'Liberation Serif', 'Georgia')
THEMES = {
    'dark': {
        'bg':'#09111b','chrome':'#0d1723','panel':'#101c29','ink':'#e5edf6','muted':'#8396ab',
        'line':'#33495a','control':'#172738','hover':'#22384d','input':'#0b1723','selected':'#18312f',
        'blue':'#73b6e7','purple':'#b4a0f1','mint':'#70cfb5','amber':'#dfb67a','pink':'#e7899d',
        'node-input':'#142536','node-linear':'#201f33','node-module':'#2a251e',
        'node-output':'#15302b','node-reshape':'#24312b','dots':'#26384b',
    },
    'light': {
        'bg':'#edf2f5','chrome':'#f7fafc','panel':'#ffffff','ink':'#172a39','muted':'#526577',
        'line':'#b9c7d1','control':'#e8eff4','hover':'#d9e5ed','input':'#f8fafc','selected':'#d8eee7',
        'blue':'#23699b','purple':'#6b45ad','mint':'#18735e','amber':'#8b5f16','pink':'#ac3354',
        'node-input':'#e0edf7','node-linear':'#ebe2f7','node-module':'#f4ead8',
        'node-output':'#dcefe7','node-reshape':'#e6eddb','dots':'#c6d3dd',
    },
    'monokai': {
        'bg':'#20211c','chrome':'#272822','panel':'#2d2e27','ink':'#f8f8f2','muted':'#babaae',
        'line':'#55574b','control':'#383a31','hover':'#46483d','input':'#20211c','selected':'#3b4628',
        'blue':'#66d9ef','purple':'#ae81ff','mint':'#a6e22e','amber':'#e6db74','pink':'#f92672',
        'node-input':'#263c3d','node-linear':'#373047','node-module':'#423c26',
        'node-output':'#304023','node-reshape':'#34412d','dots':'#414339',
    },
    'obsidian': {
        'bg':'#131217','chrome':'#1a191f','panel':'#222127','ink':'#ebe7f2','muted':'#aaa2b8',
        'line':'#48404f','control':'#302c37','hover':'#41384d','input':'#19171f','selected':'#3b3050',
        'blue':'#92b8e8','purple':'#c2a0f5','mint':'#8bcbb5','amber':'#e3bd88','pink':'#dd93b1',
        'node-input':'#222d3e','node-linear':'#30243d','node-module':'#3b3027',
        'node-output':'#213630','node-reshape':'#30362a','dots':'#37313f',
    },
    'github': {
        'bg':'#f6f8fa','chrome':'#ffffff','panel':'#ffffff','ink':'#1f2328','muted':'#59636e',
        'line':'#c9d1d9','control':'#f6f8fa','hover':'#e8edf2','input':'#ffffff','selected':'#daf0e1',
        'blue':'#0969da','purple':'#8250df','mint':'#1a7f37','amber':'#9a6700','pink':'#cf222e',
        'node-input':'#ddf4ff','node-linear':'#f0e6ff','node-module':'#fff2d5',
        'node-output':'#dafbe1','node-reshape':'#e7f1e8','dots':'#d0d7de',
    },
    'neon': {
        'bg':'#100b1d','chrome':'#171025','panel':'#21152e','ink':'#f5edff','muted':'#bdaccf',
        'line':'#57336e','control':'#2b1940','hover':'#402356','input':'#150d24','selected':'#173c3c',
        'blue':'#37e7ff','purple':'#de9dff','mint':'#4dffc1','amber':'#ffe578','pink':'#ff68b5',
        'node-input':'#102c3c','node-linear':'#311647','node-module':'#3a3020',
        'node-output':'#11372c','node-reshape':'#1e353b','dots':'#362242',
    },
    'oceanic': {
        'bg':'#071e29','chrome':'#0b2835','panel':'#103440','ink':'#e5f2f4','muted':'#92b5bf',
        'line':'#365d6c','control':'#174351','hover':'#235667','input':'#092832','selected':'#174b46',
        'blue':'#7dc9f5','purple':'#b6b0ed','mint':'#68dbc6','amber':'#e8c48d','pink':'#f5a8b0',
        'node-input':'#133a50','node-linear':'#273952','node-module':'#3a3d30',
        'node-output':'#124c44','node-reshape':'#27483d','dots':'#1e4250',
    },
    'terminal': {
        'bg':'#14120e','chrome':'#1a1711','panel':'#211c14','ink':'#ffdf9e','muted':'#c0ab86',
        'line':'#594a32','control':'#2c2418','hover':'#3b301f','input':'#15120d','selected':'#3b3018',
        'blue':'#f4c46b','purple':'#dbc5a8','mint':'#ffd376','amber':'#ffe4ac','pink':'#ff9982',
        'node-input':'#2b251b','node-linear':'#30271d','node-module':'#3b2d16',
        'node-output':'#302e19','node-reshape':'#282b1f','dots':'#393021',
    },
    'matrix': {
        'bg':'#060d08','chrome':'#09130c','panel':'#0e1c12','ink':'#c0f7c9','muted':'#7fbd8d',
        'line':'#2c5639','control':'#14291b','hover':'#20402a','input':'#08120c','selected':'#173b22',
        'blue':'#78db95','purple':'#aff0b8','mint':'#5bff8b','amber':'#d1ef93','pink':'#ffa993',
        'node-input':'#11291c','node-linear':'#1a3022','node-module':'#283221',
        'node-output':'#123b22','node-reshape':'#1e3320','dots':'#1b3522',
    },
    'math': {
        'bg':'#f5f1e7','chrome':'#fcf9f1','panel':'#fffdf6','ink':'#292b38','muted':'#656474',
        'line':'#c8c2b2','control':'#eee9dc','hover':'#e3dfd2','input':'#fffdf7','selected':'#e0e7dc',
        'blue':'#344d91','purple':'#6e4687','mint':'#376c53','amber':'#8b601e','pink':'#aa3e53',
        'node-input':'#e7edf6','node-linear':'#eee5f1','node-module':'#f5ecd7',
        'node-output':'#e2eddf','node-reshape':'#eeeddd','dots':'#d8d2c4',
    },
    'retrofuturism': {
        'bg':'#efe5cf','chrome':'#f8efd9','panel':'#fff6e1','ink':'#273f42','muted':'#656e62',
        'line':'#b6af93','control':'#e5dcc4','hover':'#d3d8c2','input':'#fff8e8','selected':'#d0e0cf',
        'blue':'#216975','purple':'#9b536c','mint':'#23745e','amber':'#986022','pink':'#ae4938',
        'node-input':'#d9e8e1','node-linear':'#eedad9','node-module':'#f0dfb9',
        'node-output':'#d6e4cc','node-reshape':'#e2e3c4','dots':'#ccc0a4',
    },
    'cyberpunk': {
        'bg':'#100e19','chrome':'#191321','panel':'#241a2c','ink':'#fff2cb','muted':'#c5b0c9',
        'line':'#60415f','control':'#352138','hover':'#4c2b48','input':'#160f20','selected':'#3e371c',
        'blue':'#63e8f7','purple':'#ec97ef','mint':'#f9ef55','amber':'#ffc17a','pink':'#ff78bb',
        'node-input':'#133544','node-linear':'#402044','node-module':'#463b1a',
        'node-output':'#3d401c','node-reshape':'#253d31','dots':'#3a2440',
    },
    'blueprint': {
        'bg':'#0b2d58','chrome':'#0d3564','panel':'#133d6b','ink':'#edf7ff','muted':'#b0cbe3',
        'line':'#537da7','control':'#1a4776','hover':'#285c8e','input':'#0b315c','selected':'#215876',
        'blue':'#bce5ff','purple':'#d6ceff','mint':'#94eedb','amber':'#ffdda1','pink':'#ffb6ce',
        'node-input':'#174d7a','node-linear':'#314875','node-module':'#3b4e61',
        'node-output':'#1b595e','node-reshape':'#29515c','dots':'#326494',
    },
    'aurora': {
        'bg':'#101724','chrome':'#162031','panel':'#1c2a38','ink':'#edf6f5','muted':'#a4bdc4',
        'line':'#405968','control':'#243b46','hover':'#335160','input':'#111e2c','selected':'#23473d',
        'blue':'#9edcf4','purple':'#daa9ef','mint':'#9cecaf','amber':'#f0d5a0','pink':'#f5adce',
        'node-input':'#233c56','node-linear':'#3d2e51','node-module':'#464231',
        'node-output':'#284b3c','node-reshape':'#304b45','dots':'#2c3c4b',
    },
    'bauhaus': {
        'bg':'#f3eee5','chrome':'#fbf7ee','panel':'#fffaf0','ink':'#272727','muted':'#62615d',
        'line':'#bdb5a9','control':'#e9e2d7','hover':'#ddd5c7','input':'#fffcf5','selected':'#eadfba',
        'blue':'#214a91','purple':'#a53732','mint':'#316452','amber':'#906500','pink':'#b32f36',
        'node-input':'#dce6f3','node-linear':'#f4ddda','node-module':'#f4e7b4',
        'node-output':'#deeadf','node-reshape':'#e9e6d5','dots':'#d0c7b9',
    },
    'metallic': {
        'bg':'#151b23','chrome':'#202832','panel':'#2a3038','ink':'#edf1f5','muted':'#abb3bd',
        'line':'#5a6470','control':'#3b444f','hover':'#4a5460','input':'#19212a','selected':'#3e4f60',
        'blue':'#8ab6d9','purple':'#c7b8e1','mint':'#add1c2','amber':'#d5b79a','pink':'#e9a9ab',
        'node-input':'#283847','node-linear':'#383342','node-module':'#403a34',
        'node-output':'#2c3d39','node-reshape':'#363d36','dots':'#3d4652',
    },
}


def resolve_theme(choice, system='dark'):
    if choice not in THEME_NAMES:
        raise ValueError('Choose a supported theme: '+', '.join(name.title() for name in THEME_NAMES)+'.')
    return ('light' if system == 'light' else 'dark') if choice == 'system' else choice


def native_palette(tokens):
    from PyQt6.QtGui import QColor, QPalette
    palette=QPalette()
    for role,key in {'Window':'chrome','WindowText':'ink','Base':'input','AlternateBase':'panel',
                     'ToolTipBase':'panel','ToolTipText':'ink','Text':'ink','Button':'control',
                     'ButtonText':'ink','BrightText':'pink','Link':'blue','Highlight':'blue',
                     'HighlightedText':'bg','PlaceholderText':'muted','Light':'hover',
                     'Mid':'line','Dark':'bg','Shadow':'bg'}.items():
        palette.setColor(getattr(QPalette.ColorRole,role),QColor(tokens[key]))
    for role in ('WindowText','Text','ButtonText'):
        palette.setColor(QPalette.ColorGroup.Disabled,getattr(QPalette.ColorRole,role),QColor(tokens['muted']))
    return palette


def native_font(theme, pixel_size, default_font):
    """Copy the baseline font so leaving Math restores the user's Qt family."""
    from PyQt6.QtGui import QFont, QFontDatabase
    font = QFont(default_font)
    font.setPixelSize(pixel_size)
    if theme == 'math':
        available = set(QFontDatabase.families())
        font.setFamilies([family for family in SERIF_FAMILIES if family in available] or ['serif'])
        font.setStyleHint(QFont.StyleHint.Serif)
    return font
