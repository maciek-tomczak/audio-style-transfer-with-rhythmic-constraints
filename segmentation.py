#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""DAFx2018 Audio Style Transfer with Rhythmic Constraints
Maciek Tomczak
"""
import numpy as np
import madmom
import scipy.io as sio

class Segmentation:
    def __init__(self,args,inputA,flnA,inputB,flnB,inputC,flnC):
        self.args =args
        self.inputA = inputA
        self.flnA = flnA
        self.inputB = inputB
        self.flnB = flnB
        self.inputC = inputC
        self.flnC = flnC
        
    def get_beats(self,fln):
        
        db_proc =madmom.features.DBNDownBeatTrackingProcessor(beats_per_bar=[4, 4], fps=100)
        dbact = madmom.features.RNNDownBeatProcessor()(self.args['audio_path']+fln)
        db = db_proc(dbact)
        
        # beats and downbeats
        Beats = [db[i,0] for i in range(len(db)) if db[i,1]!=1 or db[i,1]==1]
        bt_samples = [int(np.round(db[i,0] * self.args['sr'])) for i in range(len(db)) if db[i,1]!=1 or db[i,1]==1]
        
        if self.args['init_downbeat']: # set first beat as the downbeat and fix rest for 4/4
            n=1
            for i in range(len(db)):
                if n < 5:
                    db[i,1] = n
                    n+=1
                else:
                    n =1
        
        Downbeats = [db[i,0] for i in range(len(db)) if db[i,1]==1]
        db_samples = [int(np.round(db[i,0]*self.args['sr'])) for i in range(len(db)) if db[i,1]==1]
        
        # onsets
        onset_proc = madmom.features.onsets.OnsetPeakPickingProcessor(fps=100,threshold=0.2)
        act = madmom.features.onsets.RNNOnsetProcessor()(self.args['audio_path']+fln)
            
        onsets = onset_proc(act)
        onset_samples = [int(np.round(onsets[i]*self.args['sr'])) for i in range(len(onsets))]

        return Beats, bt_samples, Downbeats, db_samples, onsets, onset_samples, act

    def process_inputs(self): 
        """ trim inputs to the same length """
        
        if self.inputC is not None: 
            inputC_Beats, inputC_beats, inputC_Measures, inputC_measures, inputC_onsets, inputC_onsets_samples, inputC_act = self.get_beats(self.flnC)
            temp_measures=[];inputC_measures = [0 if inputC_measures[i]>len(self.inputC) else temp_measures.append(inputC_measures[i]) for i in range(len(inputC_measures)) ]; inputC_measures=temp_measures
            temp_beats=[];   inputC_beats = [0 if inputC_beats[i]>len(self.inputC) else temp_beats.append(inputC_beats[i]) for i in range(len(inputC_beats))]; inputC_beats=temp_beats
            temp_measures=[]; inputC_measures = [0 if inputC_measures[i]/float(self.args['hoplen'])>len(inputC_act) else temp_measures.append(inputC_measures[i]) for i in range(len(inputC_measures))]; inputC_measures=temp_measures
    
        inputA_Beats, inputA_beats, inputA_Measures, inputA_measures, inputA_onsets, inputA_onsets_samples, inputA_act = self.get_beats(self.flnA)

        temp_measures=[];inputA_measures = [0 if inputA_measures[i]>len(self.inputA) else temp_measures.append(inputA_measures[i]) for i in range(len(inputA_measures)) ]; inputA_measures=temp_measures
        temp_beats=[];   inputA_beats = [0 if inputA_beats[i]>len(self.inputA) else temp_beats.append(inputA_beats[i]) for i in range(len(inputA_beats))]; inputA_beats=temp_beats
        temp_pattern=[]; inputA_pattern = [0 if inputA_onsets[i]>len(self.inputA) else temp_pattern.append(inputA_onsets[i]) for i in range(len(inputA_onsets))]; inputA_pattern=temp_pattern
        temp_measures=[];inputA_measures = [0 if inputA_measures[i]/float(self.args['hoplen'])>len(inputA_act) else temp_measures.append(inputA_measures[i]) for i in range(len(inputA_measures))]; inputA_measures=temp_measures
        
        inputB_Beats, inputB_beats, inputB_Measures, inputB_measures, inputB_onsets, inputB_onsets_samples, inputB_act = self.get_beats(self.flnB)
        
        temp_measures=[];inputB_measures = [0 if inputB_measures[i]>len(self.inputB) else temp_measures.append(inputB_measures[i]) for i in range(len(inputB_measures)) ]; inputB_measures=temp_measures
        temp_beats=[];   inputB_beats = [0 if inputB_beats[i]>len(self.inputB) else temp_beats.append(inputB_beats[i]) for i in range(len(inputB_beats))]; inputB_beats=temp_beats
        temp_pattern=[]; inputB_pattern = [0 if inputB_onsets[i]>len(self.inputB) else temp_pattern.append(inputB_onsets[i]) for i in range(len(inputB_onsets))]; inputB_pattern=temp_pattern
        temp_measures=[];inputB_measures = [0 if inputB_measures[i]/float(self.args['hoplen'])>len(inputB_act) else temp_measures.append(inputB_measures[i]) for i in range(len(inputB_measures))]; inputB_measures=temp_measures        

        if self.inputC is not None:
            nummeasures = min(len(inputA_measures),len(inputB_measures),len(inputC_measures))
        else:
            nummeasures = min(len(inputA_measures),len(inputB_measures))
        
        # trim inputB
        inputB_measures=inputB_measures[:nummeasures]
        
        temp_beats=[]; inputB_beats = [inputB_beats[i] if inputB_beats[i]>inputB_measures[-1] else temp_beats.append(inputB_beats[i]) for i in range(len(inputB_beats))]; inputB_beats=temp_beats # clear beats after last downbeat
        temp_beats=[]; inputB_beats = [inputB_beats[i] if inputB_beats[i]<inputB_measures[0] else temp_beats.append(inputB_beats[i]) for i in range(len(inputB_beats))]; inputB_beats=temp_beats # clear beats before first downbeat
        temp_pattern=[]; inputB_pattern = [0 if inputB_pattern[i]>inputB_measures[-1] else temp_pattern.append(inputB_pattern[i]) for i in range(len(inputB_pattern))]; inputB_pattern=temp_pattern # clear pat events after last downbeat
        
        InputA = self.inputA[:inputA_measures[-1]]
        InputB = self.inputB[:inputB_measures[-1]]
        
        if self.inputC is not None:
            InputC = self.inputC[:inputC_measures[-1]]
        else:
            InputC = None
            inputC_measures = None
            inputC_beats = None
        
        if InputC is not None:
            n_samples =min(len(InputA),len(InputB),len(InputC))
        else:
            n_samples =min(len(InputA),len(InputB))
        
        # trim recordings to the shortest
        InputA =InputA[:n_samples]
        InputB =InputB[:n_samples]
        
        if InputC is not None: InputC =InputC[:n_samples]; assert len(InputA) == len(InputB) == len(InputC)
        assert len(InputA) == len(InputB)
        
        return InputA, InputB, InputC, inputA_measures, inputB_measures, inputC_measures, inputA_beats, inputB_beats, inputC_beats, n_samples
    
    def get_segments(self):
        """ 
        chop input into beat length segments (beat_channels)
        """
        # process_inputs can be written do be more versatile to be able to process any arbitrary input
        # right it processes all three possible inputs which makes sense but that could be simplified
        # the problem is how to process input C when it used
        # it is not obvious without additional parameter specifiying the name of the input, what actually
        # is bein processed. args.fln could actually chech that. 
        # OK, so use args.flnsA/B/C to check inside process_inputs to check what the fln is at process time...
        
        InputA, InputB, InputC, inputA_measures, inputB_measures, inputC_measures, inputA_beats, inputB_beats, inputC_beats, n_samples = self.process_inputs()
        
        n_segs = min(len(inputA_beats), len(inputB_beats)) - 1
        input_shape = (1, n_samples, 1, 1)

        count=0# input A
        xA_seg = np.ones((n_segs,np.max(np.diff(inputA_beats)) ))*1e-9
        dif = np.diff(inputA_beats)
        while (count < n_segs):
            try:
                xA_seg[count, :dif[count] ] = InputA[inputA_beats[count] : inputA_beats[count]+dif[count] ]
            except:
                dif2=InputA[inputA_beats[count] :].shape[0]
                xA_seg[count, :dif2] = InputA[inputA_beats[count] : ]
            count += 1
            
        count=0# input B
        xB_seg = np.ones((n_segs,np.max(np.diff(inputB_beats)) ))*1e-9
        dif = np.diff(inputB_beats)
        while (count < n_segs):
            try:
                xB_seg[count, :dif[count] ] = InputB[inputB_beats[count] : inputB_beats[count]+dif[count] ]
            except:
                dif2=InputB[inputB_beats[count] :].shape[0]
                xB_seg[count, :dif2] = InputB[inputB_beats[count] : ]
            count += 1
        
        if InputC is not None:# input C
            count=0
            xC_seg = np.ones((n_segs,np.max(np.diff(inputC_beats)) ))*1e-9
            dif = np.diff(inputC_beats)
            while (count < n_segs):
                try:
                    xC_seg[count, :dif[count] ] = InputC[inputC_beats[count] : inputC_beats[count]+dif[count] ]
                except:
                    dif2=InputC[inputC_beats[count] :].shape[0]
                    xC_seg[count, :dif2] = InputC[inputC_beats[count] : ]
                count += 1
        else:
            xC_seg=None
        
        # optimisable variable
        if self.args['target_pattern'] == 'A': 
            target_beats=inputA_beats
        elif self.args['target_pattern'] =='B':
            target_beats=inputB_beats
        elif self.args['target_pattern'] =='C': 
            target_beats=inputC_beats
        else:
            target_beats=inputA_beats
        count=0
        x_seg_init = np.random.standard_normal( (input_shape[1]) ).astype(np.float32)*1e-3
        x_seg = np.ones((n_segs,np.max(np.diff(target_beats)) ))*1e-9
        dif = np.diff(target_beats)
        while (count < n_segs):
            try:
                x_seg[count, :dif[count] ] = x_seg_init[target_beats[count] : target_beats[count]+dif[count] ]
            except:
                dif2=x_seg_init[target_beats[count] :].shape[0]
                x_seg[count, :dif2] = x_seg_init[target_beats[count] : ]
            count += 1
        
        beat_channels=[]
        if xC_seg is not None:    
            for i in range(n_segs):
                beat_channels.append([xA_seg[i,:],xB_seg[i,:],xC_seg[i,:],x_seg[i,:]])
        else:
            xC_seg=xA_seg
            for i in range(n_segs):
                beat_channels.append([xA_seg[i,:],xB_seg[i,:],xC_seg[i,:],x_seg[i,:]])
            
        return InputA, InputB, InputC, xA_seg, xB_seg, xC_seg, x_seg, beat_channels