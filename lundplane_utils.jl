# using ProgressMeter
using Flux, Statistics, StatsBase, GraphNeuralNetworks, Graphs, MLUtils, Plots #, CUDA
using Flux.Data: DataLoader
# using GraphMakie, CairoMakie
using Trapz, Printf
using UnROOT #Julia native package for reading root files
import Interpolations
using Random
using Glob
using BSON


function adj_list( p1ids, p2ids )
    """
    Function for constructing an adjacency list from node parent ids. 
    """
    len = (length(p1ids)>0 && length(p2ids)>0) ? maximum([maximum(p1ids), maximum(p2ids)])+1 : 0
        
    li = Vector{Vector{Int32}}()
    for i in 1:len
        el = Vector{Int32}()
        
        #find child nodes and append ids
        for ip in 1:size(p1ids)[1]
            if (p1ids[ip]+1 == i) || (p2ids[ip]+1 == i)
                push!(el,ip)
            end
        end
        
        #append parent ids
        if p1ids[i]!=-1
            push!(el,p1ids[i]+1) #Need to add +1 to indices, because Julia counts from 1!!
        end
        if p2ids[i]!=-1
            push!(el,p2ids[i]+1)
        end        
        
        push!(li,el)
    end
    return li
end


function make_graphs_from_ttree( ttree, nentries; minentry=1, lab=0.0 )
    """
    Constructs a list of input samples of the GNNGraph type from an input tree.
    """
    out_graphs = GNNGraph[]
    
    idtree = minentry
    noutsamps = 1
#     Threads.@threads for idtree in minentry:nentries #Multi-threaded loop!!!
    for idtree in minentry:nentries #Multi-threaded loop!!!
#     while noutsamps<=nentries
        if length(ttree[idtree].Akt10TruthChargedJet_jetLundIDParent1) > 0  #Skip if there are no jets            
            adj = adj_list( ttree[idtree].Akt10TruthChargedJet_jetLundIDParent1[1], 
                            ttree[idtree].Akt10TruthChargedJet_jetLundIDParent2[1] )

            dr = ttree[idtree].Akt10TruthChargedJet_jetLundDeltaR[1]
            z = ttree[idtree].Akt10TruthChargedJet_jetLundZ[1]
            kt = ttree[idtree].Akt10TruthChargedJet_jetLundKt[1]
            en = ttree[idtree].Akt10TruthChargedJet_jetLundE[1]
            pt1 = ttree[idtree].Akt10TruthChargedJet_jetLundPt1[1]
            pt2 = ttree[idtree].Akt10TruthChargedJet_jetLundPt2[1]

            
            feats = Matrix{Float32}(undef,6,length(z))
            feats[1,:] = dr
            feats[2,:] = z 
            feats[3,:] = kt 
            feats[4,:] = en
            feats[5,:] = pt1
            feats[6,:] = pt2


            if length(adj)>0
                push!( out_graphs, GNNGraph(adj, ndata=(; x = feats),  # input node features
                                                 gdata=(; y = Float32(lab), 
                                                          pt = ttree[idtree].Akt10TruthJet_jetPt[1], 
                                                          m = ttree[idtree].Akt10TruthJet_jetM[1] )) ) #labels
                noutsamps+=1
            end
        end
        idtree+=1
    end
    
    return out_graphs
end


function load_chunk_from_rootfiles( gl, nch, ich, lab=0.0 )
    fnames = glob(gl)

    brkeys = ["Akt10TruthJet_jetM", "Akt10TruthJet_jetPt",
              "Akt10TruthChargedJet_jetLundZ", "Akt10TruthChargedJet_jetLundDeltaR", "Akt10TruthChargedJet_jetLundKt",
              "Akt10TruthChargedJet_jetLundE", "Akt10TruthChargedJet_jetLundPt1", "Akt10TruthChargedJet_jetLundPt2",
              "Akt10TruthChargedJet_jetLundIDParent1", "Akt10TruthChargedJet_jetLundIDParent2"]

    tname = "lundjets_InDetTrackParticles"

    lts = [ LazyTree(ROOTFile(fn),tname,brkeys) for fn in fnames ]

    tlengths = [ length(lt) for lt in lts ]

    i_min = (ich - 1) * (sum(tlengths) ÷ nch) + 1
    i_max = ich * (sum(tlengths) ÷ nch)

    ilt_min = 1
    ilt_max = 1
    itree_min = i_min
    itree_max = i_max
    
    ltsum = 0
    for ilt in 1:length(tlengths)
        
        if i_min > sum(tlengths[1:ilt])
            ilt_min = ilt+1
            itree_min = i_min - sum(tlengths[1:ilt])
        end
        
        if i_max > sum(tlengths[1:ilt])
            ilt_max = ilt+1
            itree_max = i_max - sum(tlengths[1:ilt])
        end
    end 
    
    out_graphs = GNNGraph[]
    
    #both indices on same tree
    if ilt_min == ilt_max
        append!( out_graphs, make_graphs_from_ttree( lts[ilt_min], itree_max, minentry=itree_min, lab=lab ) )
    #indices on multiple trees
    else
        #get events on first tree
        append!( out_graphs, make_graphs_from_ttree( lts[ilt_min], tlengths[ilt_min], minentry=itree_min, lab=lab ) )
        #get events on intermediate trees
        for ilt in ilt_min+1:ilt_max-1
            append!( out_graphs, make_graphs_from_ttree( lts[ilt], tlengths[ilt], minentry=1, lab=lab ) )
        end
        #get events on last tree
        append!( out_graphs, make_graphs_from_ttree( lts[ilt_max], itree_max, minentry=1, lab=lab ) )
    end
    
    return out_graphs
end
