% interactions 2<->2  F1+F3=F2+F4

% version NMordant (moins rapide)
for if1=ID2 %boucle F1 fix
    if1
    F1(j)=if1;
    
    % boucle F2>F1 et F4>F2 car Cor4 est symetrique par echange F2 F4
    % Cor4(F4,F2,F1)
    for if2=if1:1:newN
        
        afnew3=afnew(:,2*if2-if1:end);
        afnew4=afnew(:,if2:end+if1-if2);
    
        AF1=repmat(afnew(:,if1),[1,newN-2*if2+if1+1]);
        AF2=repmat(afnew(:,if2),[1,newN-2*if2+if1+1]);
        
        Cor4(if2:(newN-if2+if1),if2,j)=Cor4(if2:(newN-if2+if1),if2,j)+ (sum(conj(afnew3.*AF1).*(afnew4.*AF2),1)).';
    end

    % boucle F2<F1
    for if2=1:if1
        
        afnew3=afnew(:,1:end-if1+if2);
        afnew4=afnew(:,if1-if2+1:end);
        
        AF1=repmat(afnew(:,if1),[1,newN-if1+if2]);
        AF2=repmat(afnew(:,if2),[1,newN-if1+if2]);
        
        Cor4(if1-if2+1:end,if2,j)=Cor4(if1-if2+1:end,if2,j)+ (sum(conj(afnew3.*AF1).*(afnew4.*AF2),1)).';
    end
    
       j=j+1;
 
end

 

