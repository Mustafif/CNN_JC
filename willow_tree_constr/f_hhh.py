import numpy as np
from scipy.stats import skew, kurtosis, norm
from typing import Tuple

def f_hhh(mean: float, sd: float, ka3: float, ka4: float, tol=1e-6) -> Tuple[int, float, float, float, float, int]: 
    if sd < 0:
        return 0, 0, 0, 0, 0, 1
    
    if sd == 0:
        return 5, 0, 0, 0, mean, 0
    
    rb1 = ka3 / (sd**3)
    b1 = rb1**2
    b2 = ka4 / (sd**4)
    
    if b2 < 0 or abs(rb1) <= tol:
        return 4, -mean / sd, 1 / sd, 1.0, 0, 0
    
    if b2 > (b1 + tol + 1):
        x = 0.5 * b1 + 1
        y = abs(rb1) * np.sqrt(0.25 * b1 + 1)
        u = (x + y)**(1/3)
        w = u + 1 / u - 1
        u = w**2 * (3 + w * (2 + w)) - 3
        
        x = u - b2
        if abs(x) > tol:
            if x > 0:
                itype, gamma, delta, xlam, xi = 3, *sbfit(mean, sd, rb1, b2, tol)
                return (itype, gamma, delta, xlam, xi, 0) if any((gamma, delta, xlam, xi)) else (4, -mean / sd, 1 / sd, 1.0, 0, 3)
            else:
                return 2, *sufit(mean, sd, rb1, b2, tol), 0
        else:
            xlam = sign(1, rb1)
            x = 1 / np.sqrt(np.log(w))
            y = 0.5 * x * np.log(w * (w - 1) / (sd**2))
            xi = xlam * (mean - np.exp((0.5 / x - y) / x))
            return 1, y, x, xlam, xi, 0
    elif b2 < (b1 + 1):
        return 0, 0, 0, 0, 0, 2
    else:
        y = 0.5 + 0.5 * np.sqrt(1 - 4 / (b1 + 4))
        y = 1 - y if rb1 > 0 else y
        x = sd / np.sqrt(y * (1 - y))
        return 5, 0, y, x, mean - y * x, 0
    


def sign(x: float, y: float) -> float: 
    return abs(x) if y >= 0 else -abs(x)

def mom(g: float, d: float) -> Tuple[list, bool]:
        zz, vv, limit = 1e-5, 1e-8, 500
        rttwo, rrtpi = np.sqrt(2), 1 / np.sqrt(2 * np.pi)
        expa, expb = 80, 23.7
        
        w = g / d
        if w > expa:
            return [], True
        
        e = np.exp(w) + 1
        r = rttwo / d
        h = 0.75 if d >= 3 else 0.25 * d
        a = [0] + [1 / e**i for i in range(1, 7)]
        
        for _ in range(1, int(limit) + 1):
            c = a.copy()
            h *= 0.5
            t = u = w
            y = h * h
            x = 2 * y
            v = y
            f = r * h
            
            for _ in range(1, int(limit) + 1):
                b = a.copy()
                u -= f
                z = 1 if u <= -expb else np.exp(u) + 1
                t += f
                l = int(t > expb)
                s = np.exp(t) + 1 if l == 0 else 0
                p = q = np.exp(-v)
                
                for i in range(1, 7):
                    p /= z
                    a[i] += p
                    if l == 0:
                        q /= s
                        a[i] += q
                        l = int(a[i] == b[i])
                    if a[i] == b[i]:
                        break
                
                y += x
                v += y
                if all(abs((a[i] - b[i]) / a[i]) <= vv for i in range(1, 7) if a[i] != 0):
                    break
            
            a = [rrtpi * h * ai for ai in a]
            if all(abs((a[i] - c[i]) / a[i]) <= zz for i in range(1, 7) if a[i] != 0):
                return a, False
    
        return [], True


def sbfit(xbar: float, sigma: float, rtb1: float, b2: float, tol: float) -> Tuple[float, float, float, float]:
    # Constants
    a = [0, 0.0124, 0.0623, 0.4043, 0.408, 0.479, 0.485, 0.5291, 0.5955, 0.626, 0.64,
         0.7077, 0.7466, 0.8, 0.9281, 1.0614, 1.25, 1.7973, 1.8, 2.163, 2.5, 8.5245, 11.346]
    
    tt = 1e-4
    limit = 50.0

    def mom(g: float, d: float) -> Tuple[list, bool]:
        # Implementation of mom function (as in the previous response)
        pass  # You need to implement this based on your requirements

    rb1 = abs(rtb1)
    b1 = rb1 * rb1
    neg = rtb1 < 0

    # Get d as a first estimate of delta
    e = b1 + 1
    x = 0.5 * b1 + 1
    y = abs(rb1) * np.sqrt(0.25 * b1 + 1)
    u = (x + y) ** (1/3)
    w = u + 1/u - 1
    f = w * w * (3 + w * (2 + w)) - 3
    e = (b2 - e) / (f - e)

    if abs(rb1) <= tol:
        f = 2
    else:
        d = 1 / np.sqrt(np.log(w))
        if d < a[10]:
            f = a[16] * d
        else:
            f = 2 - a[21] / (d * (d * (d - a[19]) + a[22]))

    f = e * f + 1
    if f < a[18]:
        d = a[13] * (f - 1)
    else:
        d = (a[9] * f - a[4]) * (3 - f) ** (-a[5])

    # Get G as first estimate of gamma
    if b1 < tt:
        g = 0
    elif d <= 1:
        g = (a[12] * d ** a[17] + a[8]) * b1 ** a[6]
    elif d <= a[20]:
        u, y = a[2], a[3]
        g = b1 ** (u * d + y) * (a[14] + d * (a[15] * d - a[11]))
    else:
        u, y = a[1], a[7]
        g = b1 ** (u * d + y) * (a[14] + d * (a[15] * d - a[11]))

    # Main iteration
    for m in range(1, int(limit) + 1):
        hmu, fault = mom(g, d)
        if fault:
            return 0, 0, 0, 0

        s = hmu[1] ** 2
        h2 = hmu[2] - s
        if h2 <= 0:
            return 0, 0, 0, 0

        t = np.sqrt(h2)
        h2a = t * h2
        h2b = h2 * h2
        h3 = hmu[3] - hmu[1] * (3 * hmu[2] - 2 * s)
        rbet = h3 / h2a
        h4 = hmu[4] - hmu[1] * (4 * hmu[3] - hmu[1] * (6 * hmu[2] - 3 * s))
        bet2 = h4 / h2b
        w = g * d
        u = d * d

        # Get derivatives
        deriv = [0] * 5
        dd = [0] * 5
        for j in range(1, 3):
            for k in range(1, 5):
                if j == 1:
                    s = hmu[k+1] - hmu[k]
                else:
                    s = ((w - k) * (hmu[k] - hmu[k+1]) + (k + 1) * (hmu[k+1] - hmu[k+2])) / u
                dd[k] = k * s / d

            t = 2 * hmu[1] * dd[1]
            s = hmu[1] * dd[2]
            y = dd[2] - t
            deriv[j] = (dd[3] - 3 * (s + hmu[2] * dd[1] - t * hmu[1]) - 1.5 * h3 * y / h2) / h2a
            deriv[j+2] = (dd[4] - 4 * (dd[3] * hmu[1] + dd[1] * hmu[3]) + 6 * (hmu[2] * t + hmu[1] * (s - t * hmu[1])) - 2 * h4 * y / h2) / h2b

        t = 1 / (deriv[1] * deriv[4] - deriv[2] * deriv[3])
        u = (deriv[4] * (rbet - rb1) - deriv[2] * (bet2 - b2)) * t
        y = (deriv[1] * (bet2 - b2) - deriv[3] * (rbet - rb1)) * t

        # Form new estimates of g and d
        g -= u
        if b1 == 0 or g < 0:
            g = 0
        d -= y
        if abs(u) <= tt and abs(y) <= tt:
            break

    delta = d
    xlam = sigma / np.sqrt(h2)
    gamma = -g if neg else g
    if neg:
        hmu[1] = 1 - hmu[1]
    xi = xbar - xlam * hmu[1]

    return gamma, delta, xlam, xi

def sufit(xbar: float, sd: float, rb1: float, b2: float, tol: float) -> Tuple[float, float, float, float]:
    b1 = rb1 * rb1
    b3 = b2 - 3

    # W is first estimate of exp(delta ** (-2))
    w = np.sqrt(np.sqrt(2 * b2 - 2.8 * b1 - 2) - 1)

    if abs(rb1) <= tol:
        # Symmetrical case - results are known
        y = 0
    else:
        # Johnson iteration (using y for his M)
        while True:
            w1 = w + 1
            wm1 = w - 1
            z = w1 * b3
            v = w * (6 + w * (3 + w))
            a = 8 * (wm1 * (3 + w * (7 + v)) - z)
            b = 16 * (wm1 * (6 + v) - b3)
            y = (np.sqrt(a * a - 2 * b * (wm1 * (3 + w * (9 + w * (10 + v))) - 2 * w1 * z)) - a) / b
            z = y * wm1 * (4 * (w + 2) * y + 3 * w1 * w1)**2 / (2 * (2 * y + w1)**3)
            v = w * w
            w = np.sqrt(np.sqrt(1 - 2 * (1.5 - b2 + (b1 * (b2 - 1.5 - v * (1 + 0.5 * v))) / z)) - 1)
            if abs(b1 - z) <= tol:
                break

        y = y / w
        y = np.log(np.sqrt(y) + np.sqrt(y + 1))
        if rb1 > 0:
            y = -y

    x = np.sqrt(1 / np.log(w))
    delta = x
    gamma = y * x
    y = np.exp(y)
    z = y * y
    x = sd / np.sqrt(0.5 * (w - 1) * (0.5 * w * (z + 1 / z) + 1))
    xlam = x
    xi = (0.5 * np.sqrt(w) * (y - 1 / y)) * x + xbar

    return gamma, delta, xlam, xi
    
