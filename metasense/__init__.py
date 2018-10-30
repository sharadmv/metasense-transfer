from . import models

BOARD_CONFIGURATION = {
    4: {
        'elcajon': {11, 12, 13},
        'donovan': {17, 19, 21},
        'shafter': {15, 18, 20},
    },
    2: {
        'elcajon': {17, 19, 21},
        'donovan': {15, 18, 20},
        'shafter': {11, 12, 13},
    },
    3: {
        'elcajon': {15, 18, 20},
        'donovan': {11, 12, 13},
        'shafter': {17, 19, 21},
    }
}
