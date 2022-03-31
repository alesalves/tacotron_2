import numpy as np
import scipy.signal as sig

def princarg(phase_in):
    '''
    This function puts an arbitrary phase value into ]-pi,pi] [rad]
    -----------------------------------------------
    '''
    phase = (phase_in+np.pi)%(-2*np.pi) + np.pi
    return phase

def robot(x, s_win=1024, n1=441, fs=None, robotFreq=None, normOrigPeak = False):
    '''
    #===== this program performs a robotization of a sound
    INPUTS
    ---------------------
    x             signal
    s_win         analysis window length [samples]
    n1            analysis step [samples]
    fs            sampling frequency (necessary only if robotFreq is informed)
    robotFreq     robot frequency [Hertz]
                  If None robotFreq will be fs/n1, otherwise n1 and n2 are ignored
    normOrigPeak  normalize according original signal max peak
    OUTPUT
    --------------------
    y             robotic signal
    '''

    
    #---- Adapting x shape to (sample, channel) ----
    if x.ndim == 1:
        DAFx_in = x.reshape((x.shape[0],1))
    elif x.ndim == 2:
        if x.shape[0]>x.shape[1]:
            DAFx_in = x.copy()
        else:
            DAFx_in = x.T.copy()
    else:
        raise TypeError('unknown audio data format !!!')
        return
    nChan = DAFx_in.shape[1]


    #----- initialize windows, arrays, etc -----
    w1  = sig.windows.hann(s_win, sym=False)   # analysis window
    w1  = np.tile(w1,nChan).reshape((nChan,len(w1))).T
    w2  = w1.copy()                            # synthesis window
    L   = DAFx_in.shape[0]

    if not(robotFreq is None):
        n1 = round(fs/robotFreq)
    n2 = n1                                    # synthesis step [samples]  ( = n1)  

    # 0-pad & normalize
    DAFx_in = np.vstack((np.zeros((s_win,nChan)),DAFx_in,np.zeros((s_win-(L%n1),nChan))))/abs(DAFx_in).max()
    DAFx_out = np.zeros(DAFx_in.shape)

    #UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU
    pin  = 0;
    pout = 0;
    pend = DAFx_in.shape[0] - s_win;
    while pin<pend:
        grain = DAFx_in[pin:pin+s_win,:] * w1;
        #===========================================
        f     = np.fft.fft(grain,axis=0); # FFT
        r     = abs(f)
        grain = np.real(np.fft.ifft(r,axis=0))*w2
        # ===========================================
        DAFx_out [pout:pout+s_win,:] = DAFx_out [pout:pout+s_win,:] + grain;
        pin  = pin + n1;
        pout = pout + n2;

    #%UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU
    #----- output -----
    #DAFx_in = DAFx_in[s_win:s_win+L,:];
    DAFx_out = DAFx_out[s_win:s_win+L,:] / abs(DAFx_out).max();
    if normOrigPeak: DAFx_out = DAFx_out * abs(x).max()

    #return DAFx_out according to original signal shape
    if x.ndim == 1:
        return DAFx_out[:,0]
    else:
        if x.shape[1] == DAFx_out.shape[1]:
            return DAFx_out
        else:
            return DAFx_out.T

def tstretch(x, n1=256, n2=512, s_win=2048, normOrigPeak = False):
    '''
    This program performs time stretching using the phase locking algorithm
    ----- user data -----
    n1            analysis step [samples]
    n2            synthesis step [samples]
    s_win         window size [samples]
    normOrigPeak  normalize according original signal max peak
    ---- output-----
    y             signal as the sum of weighted cosine
    '''
    DAFx_in = x.copy()


    #----- initialize windows, arrays, etc -----
    tstretch_ratio = n2/n1
    w1    = np.sqrt(sig.windows.hann(s_win, sym=False)) # input window
    w2    = w1.copy()    # output window
    L     = DAFx_in.shape[0]

    # 0-pad & normalize
    DAFx_in = np.concatenate([np.zeros(s_win),DAFx_in,np.zeros(s_win-(L%n1))] )/abs(DAFx_in).max()
    DAFx_out = np.zeros(s_win+int(np.ceil(DAFx_in.shape[0]*tstretch_ratio)))
    hs_win    = s_win//2
    #omegaRa = 2*np.pi*n1*np.arange(ll)/s_win; #all frequencies of FFT [0,pi[ multiplied by R_a
    #omegaRa = np.tile(omegaRa,nChan).reshape((nChan,len(omegaRa))).T
    omega = 2*np.pi*n1*np.arange(hs_win+1)/s_win
    phi0  = np.zeros(hs_win+1)
    psi   = np.zeros(hs_win+1)
    psi2 = np.zeros(hs_win+1)
    nprevpeaks=0
    
    # UUUUUUUUUUUUUUUUUUUUUUUUU
    pin =0
    pout=0
    pend= len(DAFx_in) - s_win

    while pin<pend:
        grain = DAFx_in[pin:pin+s_win].copy() * w1
        fc  = np.fft.fft(np.fft.fftshift(grain)) # FFT
        f   = fc[:hs_win+1].copy()            # positive frequency spectrum
        r   = abs(f);             # magnitudes
        phi = np.angle(f);        # phases
        peak_loc = np.zeros(hs_win)
        npeaks=0
        for b in range(2,hs_win-1):
            if (r[b]>r[b-1] and r[b]>r[b-2] and r[b]>r[b+1] and r[b]>r[b+2]):
                npeaks+=1
                peak_loc[npeaks] = b
                b = b+3
        if(pin==0):
            psi = phi.copy()
        elif(npeaks>0 and nprevpeaks>0):
            prev_p = 0
            for p in range(0,npeaks):
                p2 = int(peak_loc[p])
                while(prev_p<nprevpeaks and np.abs(p2-prev_peak_loc[prev_p+1])< np.abs(p2 - prev_peak_loc[prev_p])):
                    prev_p=prev_p+1
                p1 = int(prev_peak_loc[prev_p])
                avg_p = (p1+p2)/2
                pomega = 2*np.pi*n1*(avg_p-1)/s_win
                peak_delta_phi = pomega + princarg(phi[p2]-phi0[p1]-pomega)
                peak_target_phase = princarg(psi[p1] + peak_delta_phi*tstretch_ratio)
                peak_phase_rotation = princarg(peak_target_phase - phi[p2])
                if npeaks==1:
                    bin1 = 0
                    bin2 = hs_win + 1
                elif p==0:
                    bin1 = 0
                    bin2 = hs_win + 1
                elif p==npeaks:
                    bin1 = int(np.round((peak_loc[p-1]+p2)/2))
                    bin2 = hs_win + 1
                else:
                    bin1 = int(np.round((peak_loc[p-1]+p2)/2)) + 1
                    bin2 = int(np.round((peak_loc[p+1]+p2)/2))
                psi2[bin1:bin2] = princarg(phi[bin1:bin2] + peak_phase_rotation).copy()
            psi =psi2.copy()
        else:
            delta_phi = omega + princarg(phi-phi0 -omega)
            psi = princarg(psi+delta_phi*tstretch_ratio)
        ft = r*np.exp(1j*psi)
        ft = np.concatenate([ft,np.conjugate( np.roll(ft[::-1],1)[2:] )])
        grain = np.fft.fftshift(np.real(np.fft.ifft(ft)))*w2
        DAFx_out[pout:pout+s_win] += grain
        phi0 = phi
        prev_peak_loc = peak_loc
        nprevpeaks = npeaks
        pin = pin +n1
        pout = pout + n2
    return DAFx_out

def VX_tstretch_fft_int(x, n1=256, n2=512, s_win=2048):
    '''
    This program performs time stretching using the fft approach, when the ratio is an integer
    ----- user data -----
    n1            analysis step [samples]
    n2            synthesis step [samples]
    s_win         window size [samples]
    ---- output-----
    y             signal as the sum of weighted cosine
    '''
    DAFx_in = x.copy()


    #----- initialize windows, arrays, etc -----
    tstretch_ratio = n2/n1
    w1    = np.sqrt(sig.windows.hann(s_win, sym=False)) # input window
    w2    = w1.copy()    # output window
    L     = DAFx_in.shape[0]

    # 0-pad & normalize
    DAFx_in = np.concatenate([np.zeros(s_win),DAFx_in,np.zeros(s_win-(L%n1))] )/abs(DAFx_in).max()
    DAFx_out = np.zeros(s_win+int(np.ceil(DAFx_in.shape[0]*tstretch_ratio)))
    hs_win    = s_win//2
    #omegaRa = 2*np.pi*n1*np.arange(ll)/s_win; #all frequencies of FFT [0,pi[ multiplied by R_a
    #omegaRa = np.tile(omegaRa,nChan).reshape((nChan,len(omegaRa))).T
    omega = 2*np.pi*n1*np.arange(hs_win)/s_win
    phi0  = np.zeros(hs_win)
    psi   = np.zeros(hs_win)
    
    # UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU
    pin = 0
    pout = 0
    pend = len(DAFx_in) - s_win
    while pin<pend:
        grain = DAFx_in[pin: pin+s_win]*w1
        f = np.fft.fft(np.fft.fftshift(grain))
        r = np.abs(f)
        phi = np.angle(f)
        ft = (r* np.exp(1j*tstretch_ratio*phi))
        grain = np.fft.fftshift(np.real(np.fft.ifft(ft)))*w2
        # ===========================================
        DAFx_out[pout:pout+s_win] += grain
        pin = pin + n1
        pout = pout + n2
    return DAFx_out