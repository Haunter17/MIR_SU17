function B = preprocessQMLPnoise(Q, PFLAG)

downsamplingRate = 7;
if PFLAG == 1
    Q_Mat = log(abs(Q.c));
else
    Q_Mat = nthroot(abs(Q.c), 3);
end

QMat = Q_Mat - repmat(mean(Q_Mat, 1),size(Q_Mat,1),1);
B = QMat(:,1:downsamplingRate:size(QMat,2));

end