treebanks = {
    'ar': 'Arabic',
    'bg': 'Bulgarian',
    'ca': 'Catalan',
    'cs': 'Czech',
    'cs_cac': 'Czech-CAC',
    'cs_cltt': 'Czech-CLTT',
    'cu': 'Old_Church_Slavonic',
    'da': 'Danish',
    'de': 'German',
    'el': 'Greek',
    'en': 'English',
    'en_lines': 'English-LinES',
    'en_partut': 'English-ParTUT',
    'es': 'Spanish',
    'es_ancora': 'Spanish-AnCora',
    'et': 'Estonian',
    'eu': 'Basque',
    'fa': 'Persian',
    'fi': 'Finnish',
    'fi_ftb': 'Finnish-FTB',
    'fr': 'French',
    'fr_partut': 'French-ParTUT',
    'fr_sequoia': 'French-Sequoia',
    'ga': 'Irish',
    'gl': 'Galician',
    'gl_treegal': 'Galician-TreeGal',
    'got': 'Gothic',
    'grc': 'Ancient_Greek',
    'grc_proiel': 'Ancient_Greek-PROIEL',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hr': 'Croatian',
    'hu': 'Hungarian',
    'id': 'Indonesian',
    'it': 'Italian',
    'it_partut': 'Italian-ParTUT',
    'ja': 'Japanese',
    'kk': 'Kazakh',
    'ko': 'Korean',
    'la': 'Latin',
    'la_ittb': 'Latin-ITTB',
    'la_proiel': 'Latin-PROIEL',
    'lv': 'Latvian',
    'nl': 'Dutch',
    'nl_lassysmall': 'Dutch-LassySmall',
    'no_bokmaal': 'Norwegian-Bokmaal',
    'no_nynorsk': 'Norwegian-Nynorsk',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'pt_br': 'Portuguese-BR',
    'ro': 'Romanian',
    'ru': 'Russian',
    'ru_syntagrus': 'Russian-SynTagRus',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'sl_sst': 'Slovenian-SST',
    'sv': 'Swedish',
    'sv_lines': 'Swedish-LinES',
    'tr': 'Turkish',
    'ug': 'Uyghur',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'vi': 'Vietnamese',
    'zh': 'Chinese',
}

no_dev = {'kk', 'ug', 'uk', 'ga', 'gl_treegal', 'fr_partut', 'la', 'sl_sst'}

no_lemma = {'en_lines', 'id', 'sv_lines', 'ug', 'pt_br', 'ko'}

def path(lang, ds='train', folder="/data/ud-treebanks-conll2017/"):
    """-> str: the path for lang"""
    return "{}UD_{}/{}-ud-{}.conllu" \
        .format(folder, treebanks[lang], lang, ds)


# import src_conllu as conllu
# from collections import Counter
# for lang in treebanks:
#     freq = Counter(upos for sent in conllu.load(path(lang)) for upos in sent.upostag)
#     print("X", freq["X"], "_", freq["_"], lang, sep="\t")
