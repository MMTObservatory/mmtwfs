
def ZernikeGrad(Z, x, y, atype):

    m1, n1 = x.shape
    m2, n2 = y.shape
    if(m1 != m2 or n1 != n2):
        print('x & y are not the same size')

    if(len(Z) > 22):
        print('ZernikeGrad() is not implemented with >22 terms')
        return
    elif len(Z) < 22:
        Z = np.hstack((Z, np.zeros(22 - len(Z))))

    x2 = x * x
    y2 = y * y
    xy = x * y
    r2 = x2 + y2

    if (atype == 'dx'):
        d = Z[0] * 0. * x  # to make d an array with the same size as x
        d = d + Z[1] * 1.
        d = d + Z[2] * 0.
        d = d + Z[3] * 4. * x
        d = d + Z[4] * 2. * y
        d = d + Z[5] * 2. * x
        d = d + Z[6] * 6. * xy
        d = d + Z[7] * (9. * x2 + 3. * y2 - 2.)
        d = d + Z[8] * 6. * xy
        d = d + Z[9] * (3. * x2 - 3. * y2)
        d = d + Z[10] * 12. * x * (2. * (x2 + y2) - 1.)
        d = d + Z[11] * x * (16. * x2 - 6.)
        d = d + Z[12] * y * (24. * x2 + 8. * y2 - 6.)
        d = d + Z[13] * 4. * x * (x2 - 3. * y2)
        d = d + Z[14] * 4. * y * (3. * x2 - y2)
        d = d + Z[15] * (x2 * (50. * x2 + 60. * y2 - 36.) + y2 * (10. * y2 - 12.) + 3.)
        d = d + Z[16] * (xy * (40. * r2 - 24.))
        d = d + Z[17] * (x2 * (25. * x2 - 12. - 30. * y2) + y2 * (12. - 15. * y2))
        d = d + Z[18] * (4. * xy * (-6. + 15. * x2 + 5. * y2))
        d = d + Z[19] * 5. * (x2 * (x2 - 6. * y2) + y2 * y2)
        d = d + Z[20] * 20. * xy * (x2 - y2)
        d = d + Z[21] * 24. * x * (1. + x2 * (10. * y2 - 5. + 5. * x2) + y2 * (5. * y2 - 5.))

    elif (atype, 'dy'):

        d = Z[0] * 0. * y
        d = d + Z[1] * 0.
        d = d + Z[2] * 1.
        d = d + Z[3] * 4. * y
        d = d + Z[4] * 2. * x
        d = d + Z[5] * (-2.) * y
        d = d + Z[6] * (3. * x2 + 9. * y2 - 2.)
        d = d + Z[7] * 6. * xy
        d = d + Z[8] * (3. * x2 - 3. * y2)
        d = d + Z[9] * (-6.) * xy
        d = d + Z[10] * 12. * y * (2. * (x2 + y2) - 1.)
        d = d + Z[11] * y * (6. - 16. * y2)
        d = d + Z[12] * x * (8. * x2 + 24. * y2 - 6.)
        d = d + Z[13] * 4. * y * (y2 - 3. * x2)
        d = d + Z[14] * 4. * x * (x2 - 3. * y2)
        d = d + Z[15] * (xy * (40. * r2 - 24.))
        d = d + Z[16] * (x2 * (10. * x2 + 60. * y2 - 12.) + y2 * (50. * y2 - 36.) + 3.)
        d = d + Z[17] * (4. * xy * (6. - 5. * x2 - 15. * y2))
        d = d + Z[18] * (y2 * (-25. * y2 + 12. + 30. * x2) + x2 * (-12. + 15. * x2))
        d = d + Z[19] * 20. * xy * (y2 - x2)
        d = d + Z[20] * 5. * (x2 * (x2 - 6. * y2) + y2 * y2)
        d = d + Z[21] * 24. * y * (1. + y2 * (10. * x2 - 5. + 5. * y2) + x2 * (5. * x2 - 5.))

    return d