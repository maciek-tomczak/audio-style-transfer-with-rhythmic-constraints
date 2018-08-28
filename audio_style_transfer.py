#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""DAFx2018 Audio Style Transfer with Rhythmic Constraints
Maciek Tomczak
"""

import tensorflow as tf
import numpy as np
import librosa
import scipy.io as sio
import utils
from segmentation import Segmentation
import argparse

def compute_style_loss(x,y):
    return 2 * tf.nn.l2_loss(x - y)

def compute_content_loss(x,y,alpha=1e-2):
    return 2 * tf.nn.l2_loss(x - y) * alpha

def compute_sdif_odf(spec):
    
    diff_spec =tf.zeros_like(spec)
    diff_frames =1
    # compute flux tf
    diff_spec =spec[0,0,diff_frames:,:] - spec[0,0,0:-diff_frames,:]
    diff_spec =tf.maximum(diff_spec,0.0)
    
    return tf.reduce_sum(diff_spec,1), diff_spec[None,None,:,:]

def compute_gram(net, style_net):
    
    _, height, width, number = map(lambda i: i.value, net.get_shape())
    _, height_style, width_style, number = map(lambda i: i.value, style_net.get_shape())
    
    factor = height*width
    style_factor = height_style*width_style
    
    feats = tf.reshape(net, (-1, number))
    feats_style = tf.reshape(style_net, (-1, number))
    
    gram = tf.matmul(tf.transpose(feats), feats)/factor
    style_gram = tf.matmul(tf.transpose(feats_style), feats_style)/style_factor
    
    return gram, style_gram

def graph(args):
    
    flnA,flnB,flnC = args['inputA'],args['inputB'],args['inputC']
    typeA,wA= args['pA'][0],float(args['pA'][1])
    typeB,wB= args['pB'][0],float(args['pB'][1])
    if flnC is not None: typeC,wC= args['pC'][0],args['pC'][1]
    
    inputA, sr = librosa.load(args['audio_path']+flnA, sr=args['sr'], mono=True)
    inputB, sr = librosa.load(args['audio_path']+flnB, sr=args['sr'], mono=True)

    if flnC is not None: 
        inputC, sr = librosa.load(args['audio_path']+flnC, sr=args['sr'], mono=True)
    else: 
        inputC = None

    # segment beats
    proc = Segmentation(args,inputA,flnA,inputB,flnB,inputC,flnC)    
    inputA, inputB, inputC, xA_seg, xB_seg, xC_seg, x_seg, beat_channels = proc.get_segments()
    
    if len(x_seg[0]) // args['hoplen'] < args['k_h']: args['k_h'] = int(round(len(x_seg[0]) // args['hoplen']))
    
    n_channels = args['nfft'] // 2 + 1
    kernel_seg = utils.weight_fn( (1, args['k_h'], n_channels, args['n_filters_stft']))
    kernel_odf = utils.weight_fn( (1, 1, n_channels, args['n_filters_odf']) )
    
    dft_real_kernels, dft_imag_kernels = utils.get_stft_kernels(args['nfft'])
    
    # allocate cpu/gpu for compute
    config, devices, input_dev, seg_dev, odf_dev, output_dev = utils.get_devices()
        
    segcount=0
    result,res=[],[]
    for xA_beat_channel,xB_beat_channel,xC_beat_channel,x_beat_channel in beat_channels[:args['num_beat_segs']]:
        
        seglen =min(len(xA_beat_channel),len(xB_beat_channel),len(xC_beat_channel),len(x_beat_channel))
        print 'Transforming segment {}/{}'.format(segcount,len(beat_channels))
        segcount+=1
        
        xA_beat_channel=xA_beat_channel[:seglen]
        xB_beat_channel=xB_beat_channel[:seglen]
        xC_beat_channel=xC_beat_channel[:seglen]
        x_beat_channel = x_beat_channel[:seglen]
        
        # initalize segments
        g=tf.Graph()
        with g.as_default(), g.device(input_dev), tf.Session(config=config) as sess:
            beta_stft = tf.placeholder(tf.float32,shape=(),name='beta_stft')
            
            dft_real_kernels_tf = tf.constant(dft_real_kernels, name="dft_real_kernels", dtype='float32')
            dft_imag_kernels_tf = tf.constant(dft_imag_kernels, name="dft_imag_kernels", dtype='float32')
            
            # input A
            xA_seg_raw = np.ascontiguousarray(xA_beat_channel[None,:,None,None])
            xA_seg_raw = tf.constant(xA_seg_raw, name='xA_seg_raw', dtype='float32')
            _, xA_seg_mag, xA_seg = utils.get_logmagnitude_STFT(xA_seg_raw, dft_real_kernels_tf, dft_imag_kernels_tf, args['hoplen'])
            
            # input B
            xB_seg_raw = np.ascontiguousarray(xB_beat_channel[None,:,None,None])
            xB_seg_raw = tf.constant(xB_seg_raw, name='xB_seg_raw', dtype='float32')
            _, xB_seg_mag, xB_seg = utils.get_logmagnitude_STFT(xB_seg_raw, dft_real_kernels_tf, dft_imag_kernels_tf, args['hoplen'])
            
            # input C
            xC_seg_raw = np.ascontiguousarray(xC_beat_channel[None,:,None,None])
            xC_seg_raw = tf.constant(xC_seg_raw, name='xC_seg_raw', dtype='float32')
            _, xC_seg_mag, xC_seg = utils.get_logmagnitude_STFT(xC_seg_raw, dft_real_kernels_tf, dft_imag_kernels_tf, args['hoplen'])
            
            # optimizable variable x
            x_beat_channel = np.ascontiguousarray(x_beat_channel[None,:,None,None]).astype(np.float32)
            x_seg_raw = tf.Variable(x_beat_channel,name='x_seg_raw')
            _, x_seg_mag, x_seg = utils.get_logmagnitude_STFT(x_seg_raw, dft_real_kernels_tf, dft_imag_kernels_tf, args['hoplen'])

            # compute features
            with g.device(seg_dev):
                
                kernel_seg_tf = tf.constant(kernel_seg, name='kernel', dtype='float32')
                
                # optimizable var net
                conv = tf.nn.conv2d(x_seg,kernel_seg_tf,padding="VALID",strides=[1,1,1,1],name="conv_x_")
                net = utils.selu(conv)
                
                # net A
                segA_conv = tf.nn.conv2d(xA_seg,kernel_seg_tf,padding="VALID",strides=[1,1,1,1],name="convA_"+typeA)
                segA_net = utils.selu(segA_conv)
                
                if typeA == 'style':
                    gramA, style_gramA = compute_gram(net, segA_net)
                    segA_loss = compute_style_loss(gramA, style_gramA)
                elif typeA == 'content':
                    segA_loss = compute_content_loss(net, segA_net)
                
                # net B
                seg_B_conv =tf.nn.conv2d(xB_seg,kernel_seg_tf,padding='VALID',strides=[1,1,1,1],name='convB_'+typeB)            
                segB_net = utils.selu(seg_B_conv)
                
                if typeB == 'style':
                    gramB, style_gramB = compute_gram(net, segB_net)
                    segB_loss = compute_style_loss(gramB, style_gramB)
                elif typeB == 'content':
                    segB_loss = compute_content_loss(net, segB_net)
                
                # net C
                if inputC is not None:
                    segC_conv=tf.nn.conv2d(xC_seg,kernel_seg_tf,padding="VALID",strides=[1,1,1,1],name="convC_"+typeC)
                    segC_net = utils.selu(segC_conv)
                    
                    if typeC == 'style':
                        gramC, style_gramC = compute_gram(net, segC_net)
                        segC_loss = compute_style_loss(gramC, style_gramC)
                    elif typeC == 'content':
                        segC_loss = compute_content_loss(net, segC_net)
                else:
                    segC_loss=None
    
                grads_segA = tf.gradients(segA_loss, x_seg_raw)[0]
                norm_segA = tf.norm(grads_segA)
    
                grads_segB = tf.gradients(segB_loss, x_seg_raw)[0]
                norm_segB = tf.norm(grads_segB)
    
                if inputC is not None:
                    grads_segC = tf.gradients(segC_loss, x_seg_raw)[0]
                    norm_segC = tf.norm(grads_segC)
              
            if args['mode']=='ODF':
                # cosine distance loss from spectral difference evelopes
                with g.device(odf_dev):
                    sdif_odf_loss =tf.ones(())
                    odf_conv = tf.nn.conv2d(x_seg,kernel_odf,padding="VALID",strides=[1,1,1,1],name="conv_"+'odf')
                    odf_net = utils.selu(odf_conv)
                    
                    x_odf,x_sdif = compute_sdif_odf(odf_net)
                    
                    if args['target_odf_pattern'] == 'A':
                        _odf,_spec =compute_sdif_odf(xA_seg)
                    elif args['target_odf_pattern'] == 'B':
                        _odf,_spec =compute_sdif_odf(xB_seg)
                    elif args['target_odf_pattern'] == 'C':
                        _odf,_spec =compute_sdif_odf(xC_seg)
                    else:
                        _odf,_spec =compute_sdif_odf(xA_seg)
                    
                    sdif_odf_loss = tf.losses.cosine_distance(x_odf,_odf,dim=0)
            else:
                sdif_odf_loss =0
                
            # compute outputs
            with g.device(output_dev):
                total_inputA_loss=0
                total_inputB_loss=0
                total_inputC_loss=0
                total_odf_loss = sdif_odf_loss
                total_inputA_loss = wA * segA_loss
                total_inputB_loss = wB * segB_loss
                if inputC is not None: total_inputC_loss = float(wC) *segC_loss
                
                total_loss = total_inputA_loss + total_inputB_loss + total_inputC_loss + total_odf_loss
                
                with tf.Session(config=config) as sess:    
                    sess.run(tf.global_variables_initializer())
                    feed_dict ={beta_stft: 1.0}
                    
                    segA = norm_segA.eval(feed_dict=feed_dict)
                    segB = norm_segB.eval(feed_dict=feed_dict)
                    
                    if inputC is not None: segC = norm_segC.eval(feed_dict=feed_dict)
                    # normalize loss gradient
                    beta_vals =1.0
                    beta_vals =segA/segB
                    if inputC is not None: beta_vals =abs(segA+segB+segC)/3.0
                    feed_dict[beta_stft]=beta_vals 
                    
                    # train optimizer
                    opt = tf.contrib.opt.ScipyOptimizerInterface(total_loss,var_list=[x_seg_raw],method='L-BFGS-B',options={'maxiter': args['iterations'],'ftol': args['factor']* np.finfo(float).eps,'gtol': args['factor']* np.finfo(float).eps},tol=args['factor'])
                    opt.minimize(sess, feed_dict=feed_dict)
                    res = x_seg_raw.eval()
                    print 'Segment loss:', total_loss.eval(feed_dict=feed_dict)
            
        result.append(res.squeeze())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-A', '--inputA', type=str, help='filename of input A', required=True)
    parser.add_argument('-B', '--inputB', type=str, help='filename of input B', required=True)
    parser.add_argument('-C', '--inputC', type=str, help='filename of input C', default=None)
    parser.add_argument('-pA', '--pA', nargs='*', type=str, help='content or style loss',default='style 0.5')
    parser.add_argument('-pB', '--pB', nargs='*', type=str, help='content or style loss',default='style 0.5')
    parser.add_argument('-pC', '--pC', nargs='*', type=str, help='content or style loss',default='content 1.0')
    parser.add_argument('-audio_path','--audio_path',type=str, help='path to audio inputs',default='./')
    parser.add_argument('-odir', '--outdir', type=str, help='transformation output directory', default='./')
    parser.add_argument('-ofln', '--outfln', type=str, help='transformation output filename', default='output.wav')
    parser.add_argument('-sr', '--sr', type=int, help='sample rate', default=22050)
    parser.add_argument('-nfft','--nfft', type=int, help='number of FFT in samples', default=2048)
    parser.add_argument('-hoplen', '--hoplen', type=int, help='hop length in samples', default=1024)
    parser.add_argument('-n_filters_stft', '--n_filters_stft', type=int, help='stft loss filter number', default=4096)
    parser.add_argument('-n_filters_odf', '--n_filters_odf', type=int, help='odf loss filter number', default=1025)
    parser.add_argument('-k_h', '--k_h', help='filter height', type=int, default=16)
    parser.add_argument('-iters','--iterations', type=int, help='optimisaiton iterations', default=300)
    parser.add_argument('-factor','--factor', type=int, help='L-BFGS-B fctr parameter', default=1e9)
    parser.add_argument('-num_beat_segs','--num_beat_segs', type=int, help='number of transformed beat segments (default uses all segments)', default=-1)
    parser.add_argument('-init_downbeat','--init_downbeat', type=bool, help='set first detected beat to first downbeat', default=False)
    parser.add_argument('-target_pattern','--target_pattern', type=str, help='target output beat length from files: \'A\',\'B\' or \'C\'', default='B')
    parser.add_argument('-target_odf_pattern','--target_odf_pattern', type=str, help='target length of the odf pattern: \'A\',\'B\' or \'C\'', default='B')
    parser.add_argument('-m','--mode',type=str,help='mode for [normal] or \'ODF\' loss formulation',default='normal')
    
    args = vars(parser.parse_args())

    result = graph(args)
    
    result = np.concatenate([result[i] for i in range(len(result))])
    result_norm = result/np.abs(result).max()
    
    librosa.output.write_wav(args['outdir']+args['outfln']+'.wav',result_norm,sr=args['sr'],norm=False)