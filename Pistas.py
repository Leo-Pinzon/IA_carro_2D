import numpy as np
def pista(n):
    if n == '1':
        #PISTA 1
        pista_a = np.array([ #superior esquerdo
                    [0, 220],
                    [50, 330],
                    [180, 400],
                    [0, 400],

                ])
        pista_b = np.array([ #inferior esquerdo
                    [0, 180],
                    [50, 70],
                    [180, 0],
                    [0, 0],
                ])

        pista_c = np.array([ #inferior central
                    [300, 0],
                    [683, 100],
                    [1066, 0],
                    [300, 0],
                ])

        pista_d = np.array([ #superior central
                    [300, 400],
                    [683, 300],
                    [1066, 400],
                    [300, 400],
                ])

        pista_e = np.array([ #central
                    [160, 100],
                    [100, 200],
                    [160, 300],
                    [700, 200],
                ])

        pista_f = np.array([
                    [1366, 60],
                    [1366, 100],
                    [1366, 340],
                    [600, 200],
                ])

        pista_g = np.array([
                    [1366, 400],
                    [1366, 750],
                    [0, 750],
                    [0, 400],
                ])

        #linha de chegada
        check_box = np.array([
                    [1366, 400],
                    [1366, 300],
                    [1166, 300],
                    [1166, 400],
                ])
        start_point = [1200, 50]

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - PISTA 2
    if n == '2':
        #PISTA 2
        pista_a = np.array([  # superior esquerdo
            [0, 220],
            [50, 330],
            [180, 400],
            [0, 400],

        ])
        pista_b = np.array([  # inferior esquerdo
            [0, 180],
            [50, 70],
            [180, 0],
            [0, 0],
        ])

        pista_c = np.array([  # inferior central
            [640, 0],
            [726, 0],
            [696, 20],
            [670, 20],
        ])

        pista_d = np.array([  # superior direito
            [1366, 220],
            [1316, 330],
            [1186, 400],
            [1366, 400],
        ])

        pista_e = np.array([  # central
            [160, 100],
            [160, 300],
            [1206, 300],
            [1206, 100],
        ])

        pista_f = np.array([  #central direito
            [1366, 180],
            [1316, 70],
            [1186, 0],
            [1366, 0],
        ])

        pista_g = np.array([
            [1366, 400],
            [1366, 750],
            [0, 750],
            [0, 400],
        ])

        # linha de chegada
        check_box = np.array([
            [500, 400], #superior direito
            [500, 200],
            [450, 200], #inferior esquerdo
            [450, 400],
        ])
        start_point = [370, 350]

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - PISTA 3
    if n == '3':
        # PISTA 3
        pista_a = np.array([  # superior esquerdo
            [0, 220],
            [50, 330],
            [180, 400],
            [0, 400],

        ])
        pista_b = np.array([  # inferior esquerdo
            [0, 180],
            [50, 70],
            [180, 0],
            [0, 0],
        ])

        pista_c = np.array([
            [640, 400],
            [726, 400],
            [696, 380],
            [670, 380],
        ])

        pista_d = np.array([  # superior direito
            [1366, 220],
            [1316, 330],
            [1186, 400],
            [1366, 400],
        ])

        pista_e = np.array([  # central
            [160, 100],
            [160, 300],
            [1206, 300],
            [1206, 100],
        ])

        pista_f = np.array([  # central direito
            [1366, 180],
            [1316, 70],
            [1186, 0],
            [1366, 0],
        ])

        pista_g = np.array([
            [1366, 400],
            [1366, 750],
            [0, 750],
            [0, 400],
        ])

        # linha de chegada
        check_box = np.array([
            [500, 200],  # superior direito
            [500, 000],
            [450, 000],  # inferior esquerdo
            [450, 200],
        ])
        start_point = [370, 50]



    return pista_a, pista_b, pista_c, pista_d, pista_e, pista_f, pista_g, check_box, start_point