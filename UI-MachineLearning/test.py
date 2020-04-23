import docx

print('generate docx')
filename = 'files/comment.docx'
d = docx.Document(filename)

for para in d.paragraphs:
    print(para)

print('Done.')
