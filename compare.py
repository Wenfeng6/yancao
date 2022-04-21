# import face_recognition
# import numpy as np

# def match(face_encodings, face_to_compare, margin=0.6):
#     dist_matrix = face_recognition.face_distance(face_encodings, face_to_compare)

#     dist_matrix = (margin - dist_matrix) / margin
    
#     dist_matrix = np.clip(dist_matrix, 0., 1.)

#     return dist_matrix

import face_recognition
import numpy as np


def cos_similarity(x, y):
    x = np.stack(x)
    num = (x * y).sum(axis=1)
    denom = np.linalg.norm(x, axis=1) * np.linalg.norm(y)
    cos = num / denom
    return cos

def match(face_encodings, face_to_compare, margin=15., mode=1):
    dist_matrix = cos_similarity(face_encodings, face_to_compare)
    print(np.sort(dist_matrix)[::-1])
    print(dist_matrix.shape)
    # dist_matrix = face_recognition.face_distance(face_encodings, face_to_compare)
    # print(np.linalg.norm(face_encodings - face_to_compare, axis=1))
    # print(mode)
    # print(f'dist_matrix:{dist_matrix}')
    # print(dist_matrix.min())
    # print(dist_matrix.max())

    match = int(np.argmin(dist_matrix))
    match_dist = dist_matrix[match]

    result = np.zeros_like(dist_matrix)

    if mode == 1:
        for i in range(result.shape[0]):
            if dist_matrix[i] > 0.45:
                result[i] = dist_matrix[i] * 0.5 + 0.5

    elif mode == 2:
        for i in range(result.shape[0]):
            if dist_matrix[i] > 0.42:
                result[i] = dist_matrix[i] * 0.5 + 0.5
            
    # if mode == 1:
    #     margin = 22.
    #     if match_dist < margin:
    #         for i in range(result.shape[0]):
    #             if dist_matrix[i] - match_dist < 3. and dist_matrix[i] < margin:
    #                 result[i] = (margin - dist_matrix[i]) / margin * 0.2 + 0.8

    # elif mode == 2:
    #     margin = 26.5
    #     for i in range(result.shape[0]):
    #         if dist_matrix[i] < margin:
    #             result[i] = (margin - dist_matrix[i]) / margin * 0.2 + 0.8

        
    # elif mode == 2:
    #     margin = 27.
    #     delta = 3.
    #     rank_list = np.argsort(dist_matrix)
    #     i = 0
    #     print(dist_matrix[rank_list[i]])
    #     if dist_matrix[rank_list[i]] < margin:
    #         result[rank_list[i]] = (margin - dist_matrix[rank_list[i]]) / margin * 0.2 + 0.8
    #     for i in range(1, result.shape[0]):
    #         tmp = dist_matrix[rank_list[i]] - dist_matrix[rank_list[i-1]]
    #         print(dist_matrix[rank_list[i]])
    #         print(tmp)
    #         if dist_matrix[rank_list[i]] < margin and tmp < delta * 1.2:
    #             result[rank_list[i]] = (margin - dist_matrix[rank_list[i]]) / margin * 0.2 + 0.8
    #             delta = min(delta, tmp)
    #         else:
    #             break

        # match_dist = (margin - match_dist) / margin
        # result[match] = match_dist 

    # dist_matrix = (margin - dist_matrix) / margin
    # dist_matrix = 1 / (1 + np.exp( -(dist_matrix - 0.12) * 3) )
    
    # dist_matrix = np.clip(dist_matrix, 0., 1.)

    # dist_matrix = 0.4 + 0.6 * dist_matrix

    # print(dist_matrix)
    # dist_matrix = np.sqrt(dist_matrix)
    # dist_matrix = np.arctan(dist_matrix) / np.arctan(1)
    # print(result)
    return result

