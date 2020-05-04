from ml_models.pgm import CRF

x = [
    [1, 2, 3, 0, 1, 3, 4],
    [1, 2, 3],
    [0, 2, 4, 2],
    [4, 3, 2, 1],
    [3, 1, 1, 1, 1],
    [2, 1, 3, 2, 1, 3, 4]
]
y = x

crf = CRF(output_status_num=5, input_status_num=5)
crf.fit(x, y)
print(crf.predict(x[-1]))
print(crf.predict([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 3, 1, 2]))
print(len(crf.FF.feature_funcs))
