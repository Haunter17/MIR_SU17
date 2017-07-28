function B = preprocessQMLPnormc(Q, PFLAG)

downsamplingRate = 7;
if PFLAG == 1
    Q_Mat = log(abs(Q.c));
else
    Q_Mat = nthroot(abs(Q.c), 3);
end

QMat = normc(Q_Mat);
B = QMat(:,1:downsamplingRate:size(QMat,2));

end