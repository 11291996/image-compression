#importing the image 
from imageio import imread
image = imread('https://github.com/engineersCode/EngComp4_landlinear/blob/master/images/washington-monument.jpg?raw=true')
#check info 
print(type(image))
print(image.shape)
print(image.dtype)
#show image 
pyplot.figure(figsize=(2,2))
pyplot.imshow(image, cmap='gray')
#do SVD 
U, S, VT = np.linalg.svd(image)
print(U.shape, S.shape, VT.shape)
#check singular values 
pyplot.figure(figsize=(2,2))
pyplot.scatter(numpy.arange(len(S)), S, s=0.5)
pyplot.yscale('log')
pyplot.ylabel('singular value')
#how many values will be used 
from ipywidgets import widgets, fixed
k_slider = widgets.IntSlider(min=1, max=149, step=5)
display(k_slider)
#do low rank approximation from linear algebra using python 
def approximate(k, u, s, vt, image):

    #k singular value U,S,V
    u_k = u[:,:k]
    s_k = s[:k]
    vt_k = vt[:k,:]
    copy = numpy.round(u_k @ numpy.diag(s_k) @ vt_k)

    #low-rank matrix and original difference 
    diff = image - copy

    fig = pyplot.figure(figsize=(4,2))
    ax1 = pyplot.subplot(121)
    ax1.imshow(copy, cmap='gray')
    ax1.set_title('compressed image'.format(k))
    ax2 = pyplot.subplot(122)
    ax2.imshow(image, cmap='gray')
    ax2.set_title('original image'.format(k))
    
    #ratio
    ratio = image.size / (u_k.size + s_k.size + vt_k.size)
    norm = numpy.linalg.norm(diff)
    pyplot.subplots_adjust(top=0.85)
    fig.suptitle('$k = {}$, compression ratio = {:.2f}, norm = {:.2f}'
                 .format(k, ratio, norm), fontsize=5)
#slider to change SVD 
widgets.interact(approximate, k=k_slider, u=fixed(U),
                 s=fixed(S), vt=fixed(VT), image=fixed(image));