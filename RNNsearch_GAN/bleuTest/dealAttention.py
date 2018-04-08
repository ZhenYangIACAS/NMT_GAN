#!/usr/bin/python
import sys

def main(source, target, alpha, outfile='Alpha_dealed.txt'):
    f_source = open(source, 'r')
    f_target = open(target, 'r')
    f_alpha = open(alpha, 'r')
    f_out = open(outfile, 'w')
    
    sources=[source.strip()+" eos" for source in f_source.readlines()]
    targets=[target.strip()+" eos" for target in f_target.readlines()]

    alphas=[alpha.strip('\n\[\]') for alpha in f_alpha.readlines() if alpha.find('#')]
    
    alphaLists=[alpha.split('#') for alpha in alphas]

    f_source.close()
    f_target.close()
    f_alpha.close()

    assert(len(sources)==len(targets)==len(alphaLists))

    for i, source in enumerate sources:
        f_out.write(source+'\n')
        f_out.write(targets[i]+'\n')

        for alp in alphaLists[i]:
            f_out.write(str(alp)+'\n')

    f_out.close()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])


