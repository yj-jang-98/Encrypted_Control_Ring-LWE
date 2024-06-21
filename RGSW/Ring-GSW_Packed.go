// package main

// import (
// 	"fmt"
// 	"math"
// 	"os"
// 	"time"

// 	// "reflect"

// 	"github.com/tuneinsight/lattigo/v4/rgsw"
// 	"github.com/tuneinsight/lattigo/v4/ring"
// 	"github.com/tuneinsight/lattigo/v4/rlwe"
// 	"github.com/tuneinsight/lattigo/v4/rlwe/ringqp"
// 	"gonum.org/v1/plot"
// 	"gonum.org/v1/plot/plotter"

// 	"gonum.org/v1/plot/vg"
// 	"gonum.org/v1/plot/vg/draw"
// 	"gonum.org/v1/plot/vg/vgimg"
// )

// func modZq(a [][]float64, params rlwe.Parameters) [][]float64 {
// 	// Components of the matrix 'a' belongs to [-q/2, q/2)
// 	// Takes the modulo operation and maps all components to [0,q)

// 	q := float64(params.Q()[0])
// 	b := make([][]float64, len(a))
// 	for i := 0; i < len(a); i++ {
// 		b[i] = make([]float64, len(a[0]))
// 		for j := 0; j < len(a[0]); j++ {
// 			b[i][j] = a[i][j] - math.Floor(a[i][j]/q)*q
// 		}
// 	}
// 	return b
// }

// func modZqVec(a []float64, params rlwe.Parameters) []float64 {
// 	// Components of the vector 'a' belongs to [-q/2, q/2)
// 	// Takes the modulo operation and maps all components to [0,q)

// 	q := float64(params.Q()[0])
// 	b := make([]float64, len(a))
// 	for i := 0; i < len(a); i++ {
// 		b[i] = a[i] - math.Floor(a[i]/q)*q
// 	}
// 	return b
// }

// func externalProduct(ctB []*rlwe.Ciphertext, ctA []*rgsw.Ciphertext, evaluatorRGSW *rgsw.Evaluator, ringQ *ring.Ring, params rlwe.Parameters) *rlwe.Ciphertext {
// 	// Computes the external product between ctA and ctB
// 	// ctA: n-dimensional RLWE ciphertexts vector
// 	//// Each element is an RLWE encryption of each elements of vector A
// 	// ctB: n-dimensional RGSW ciphertexts vector
// 	//// Each element is an RGSW encryption of each columns of matrix B
// 	// ctC: 1-dimensional RLWE (packed) ciphertext

// 	row := len(ctA)
// 	packedCtC := rlwe.NewCiphertext(params, ctB[0].Degree(), ctB[0].Level())
// 	tmpCt := rlwe.NewCiphertext(params, ctB[0].Degree(), ctB[0].Level())
// 	for i := 0; i < row; i++ {
// 		evaluatorRGSW.ExternalProduct(ctB[i], ctA[i], tmpCt)
// 		ringQ.Add(packedCtC.Value[0], tmpCt.Value[0], packedCtC.Value[0])
// 		ringQ.Add(packedCtC.Value[1], tmpCt.Value[1], packedCtC.Value[1])
// 	}

// 	return packedCtC
// }

// func unpackPackedCt(packedCtC *rlwe.Ciphertext, n int, tau int, evaluatorRLWE *rlwe.Evaluator, ringQ *ring.Ring, params rlwe.Parameters) []*rlwe.Ciphertext {
// 	// Unpacks a packed ciphertext and returns an n-dimensional RLWE ciphertexts vector

// 	scalar := params.Q()[0] - uint64((params.Q()[0]+1)/uint64(tau))
// 	ctUnpack := make([]*rlwe.Ciphertext, tau)
// 	ctOut := make([]*rlwe.Ciphertext, n)
// 	for i := 0; i < tau; i++ {
// 		ctUnpack[i] = rlwe.NewCiphertext(params, packedCtC.Degree(), packedCtC.Level())
// 	}
// 	tmpCt := rlwe.NewCiphertext(params, packedCtC.Degree(), packedCtC.Level())

// 	// Scaling
// 	ringQ.MulScalar(packedCtC.Value[0], scalar, packedCtC.Value[0])
// 	ringQ.MulScalar(packedCtC.Value[1], scalar, packedCtC.Value[1])

// 	ctUnpack[0] = packedCtC

// 	for i := tau; i > 1; i /= 2 {
// 		for j := 0; j < tau; j += i {
// 			evaluatorRLWE.Automorphism(ctUnpack[j], uint64(i+1), tmpCt)

// 			ringQ.Sub(tmpCt.Value[0], ctUnpack[j].Value[0], ctUnpack[i/2+j].Value[0])
// 			ringQ.Sub(tmpCt.Value[1], ctUnpack[j].Value[1], ctUnpack[i/2+j].Value[1])

// 			ringQ.Add(ctUnpack[j].Value[0], tmpCt.Value[0], ctUnpack[j].Value[0])
// 			ringQ.Add(ctUnpack[j].Value[1], tmpCt.Value[1], ctUnpack[j].Value[1])

// 			ringQ.InvNTT(ctUnpack[i/2+j].Value[0], ctUnpack[i/2+j].Value[0])
// 			ringQ.InvNTT(ctUnpack[i/2+j].Value[1], ctUnpack[i/2+j].Value[1])

// 			ringQ.MultByMonomial(ctUnpack[i/2+j].Value[0], params.N()-params.N()/i, ctUnpack[i/2+j].Value[0])
// 			ringQ.MultByMonomial(ctUnpack[i/2+j].Value[1], params.N()-params.N()/i, ctUnpack[i/2+j].Value[1])

// 			ringQ.NTT(ctUnpack[i/2+j].Value[0], ctUnpack[i/2+j].Value[0])
// 			ringQ.NTT(ctUnpack[i/2+j].Value[1], ctUnpack[i/2+j].Value[1])
// 		}
// 	}

// 	// Bit reverse
// 	j := 0
// 	for i := 1; i < tau; i += 1 {
// 		bit := tau >> 1
// 		for j >= bit {
// 			j -= bit
// 			bit >>= 1
// 		}
// 		j += bit
// 		if i < j {
// 			ctUnpack[i], ctUnpack[j] = ctUnpack[j], ctUnpack[i]
// 		}
// 	}

// 	// Takes the first n ciphertexts
// 	for j := 0; j < n; j += 1 {
// 		ctOut[j] = ctUnpack[j].CopyNew()
// 	}

// 	return ctOut
// }

// func encryptRlwe(A []float64, scale float64, encryptorRLWE rlwe.Encryptor, ringQ *ring.Ring, params rlwe.Parameters) []*rlwe.Ciphertext {
// 	// Encrypts an n-dimensional float vector A into an n-dimensional RLWE ciphertexts vector ctA after scaling

// 	row := len(A)
// 	ctA := make([]*rlwe.Ciphertext, row)
// 	A_ := scalarVecMult(scale, A)
// 	modA := modZqVec(A_, params)
// 	for r := 0; r < row; r++ {
// 		pt := rlwe.NewPlaintext(params, params.MaxLevel())
// 		pt.Value.Coeffs[0][0] = uint64(modA[r])
// 		ringQ.NTT(pt.Value, pt.Value)
// 		ctA[r] = encryptorRLWE.EncryptNew(pt)
// 	}

// 	return ctA
// }

// func encryptRgsw(A [][]float64, tau int, encryptorRGSW *rgsw.Encryptor, levelQ int, levelP int, decompRNS int, decompPw2 int, ringQ *ring.Ring, ringQP *ringqp.Ring, params rlwe.Parameters) []*rgsw.Ciphertext {
// 	// Encrypts an m-by-n-dimensional float matrix A into an n-dimensional RGSW ciphertexts vector ctA

// 	row := len(A)
// 	col := len(A[0])
// 	ctA := make([]*rgsw.Ciphertext, col)
// 	modA := modZq(A, params)
// 	for c := 0; c < col; c++ {
// 		pt := rlwe.NewPlaintext(params, params.MaxLevel())
// 		for j := 0; j < tau; j++ {
// 			if j < row {
// 				// Store in the packing slots
// 				pt.Value.Coeffs[0][params.N()*j/tau] = uint64(modA[j][c])
// 			} else {
// 				pt.Value.Coeffs[0][params.N()*j/tau] = uint64(0)
// 			}
// 		}
// 		ringQ.NTT(pt.Value, pt.Value)
// 		ctA[c] = rgsw.NewCiphertext(levelQ, levelP, decompRNS, decompPw2, *ringQP)
// 		encryptorRGSW.Encrypt(pt, ctA[c])
// 	}
// 	return ctA
// }

// func decryptNewRlwe(ctA []*rlwe.Ciphertext, decryptorRLWE rlwe.Decryptor, scale float64, ringQ *ring.Ring, params rlwe.Parameters) []float64 {
// 	// 1) Decrypts an n-dimensional RLWE vector ctA and obtain an n-dimensional integer vector pt
// 	// 2) Maps the constant terms of pt from the set [0,q/2) back to [-q/2, q/2)
// 	// 3) Scale down and return decA

// 	row := len(ctA)
// 	q := float64(params.Q()[0])
// 	offset := uint64(q / (scale * 2.0))
// 	decA := make([]float64, row)
// 	for r := 0; r < row; r++ {
// 		ringQ.AddScalar(ctA[r].Value[0], offset, ctA[r].Value[0])
// 		pt := decryptorRLWE.DecryptNew(ctA[r])
// 		if pt.IsNTT {
// 			params.RingQ().InvNTT(pt.Value, pt.Value)
// 		}
// 		ringQ.SubScalar(ctA[r].Value[0], offset, ctA[r].Value[0])
// 		// Constant terms
// 		val := float64(pt.Value.Coeffs[0][0])
// 		// Mapping to [-q/2, q/2)
// 		val = val - math.Floor((val+q/2.0)/q)*q
// 		// Scale down
// 		decA[r] = val * scale
// 	}
// 	return decA
// }

// func matVecMult(A [][]float64, B []float64) []float64 {
// 	// A : m x n
// 	// B : n x l
// 	m := len(A)
// 	n := len(A[0])
// 	n1 := len(B)

// 	if n != n1 {
// 		panic(fmt.Errorf("Matrix dimension don't match"))
// 	}

// 	C := make([]float64, m)

// 	for i := 0; i < m; i++ {
// 		tmp := 0.0
// 		for k := 0; k < n; k++ {
// 			tmp = tmp + A[i][k]*B[k]
// 		}
// 		C[i] = tmp
// 	}
// 	return C
// }

// func vecAdd(A []float64, B []float64) []float64 {
// 	// A : m x 1
// 	// B : m x 1
// 	m := len(A)
// 	C := make([]float64, m)

// 	for i := 0; i < m; i++ {
// 		C[i] = A[i] + B[i]
// 	}
// 	return C
// }

// func ctAdd(ctA *rlwe.Ciphertext, ctB *rlwe.Ciphertext, params rlwe.Parameters) *rlwe.Ciphertext {
// 	// A : m x n
// 	// B : m x n
// 	ctC := rlwe.NewCiphertext(params, ctB.Degree(), ctB.Level())

// 	params.RingQ().Add(ctA.Value[0], ctB.Value[0], ctC.Value[0])
// 	params.RingQ().Add(ctA.Value[1], ctB.Value[1], ctC.Value[1])

// 	return ctC
// }

// func vec2norm(v []float64) float64 {
// 	tmp := 0.0
// 	for i := range v {
// 		tmp = tmp + v[i]*v[i]
// 	}
// 	return math.Sqrt(tmp)
// }

// func vecSubtract(v1 []float64, v2 []float64) []float64 {
// 	vReturn := make([]float64, len(v1))
// 	for i := range v1 {
// 		vReturn[i] = v1[i] - v2[i]
// 	}
// 	return vReturn
// }

// func scalarMatMult(s float64, M [][]float64) [][]float64 {
// 	C := make([][]float64, len(M))
// 	for i := 0; i < len(M); i++ {
// 		C[i] = make([]float64, len(M[0]))
// 		for j := range M[i] {
// 			C[i][j] = s * M[i][j]
// 		}
// 	}
// 	return C
// }

// func scalarVecMult(s float64, V []float64) []float64 {
// 	C := make([]float64, len(V))
// 	for i := 0; i < len(V); i++ {
// 		C[i] = s * V[i]
// 	}
// 	return C
// }

// func roundVec(M []float64) []float64 {
// 	C := make([]float64, len(M))
// 	for i := range M {
// 		C[i] = math.Round(M[i])
// 	}
// 	return C
// }

// func main() {
// 	params, _ := rlwe.NewParametersFromLiteral(rlwe.ParametersLiteral{
// 		LogN:           11,
// 		LogQ:           []int{54},
// 		Pow2Base:       7,
// 		DefaultNTTFlag: true,
// 	})
// 	fmt.Println("Degree N:", params.N())
// 	fmt.Println("Ciphertext modulus Q:", params.QBigInt(), "some prime close to 2^54")

// 	kgen := rlwe.NewKeyGenerator(params)
// 	sk := kgen.GenSecretKey()
// 	rlk := kgen.GenRelinearizationKey(sk, 1)

// 	// ======== Compute tau!! ========
// 	// least power of two greater than n, p_, and m
// 	tau := 8

// 	// Generate DFS index
// 	dfsId := make([]int, tau)
// 	for i := 0; i < tau; i++ {
// 		dfsId[i] = i
// 	}

// 	tmp := make([]int, tau)
// 	for i := 1; i < tau; i *= 2 {
// 		id := 0
// 		currBlock := tau / i
// 		nextBlock := currBlock / 2
// 		for j := 0; j < i; j++ {
// 			for k := 0; k < nextBlock; k++ {
// 				tmp[id] = dfsId[j*currBlock+2*k]
// 				tmp[nextBlock+id] = dfsId[j*currBlock+2*k+1]
// 				id++
// 			}
// 			id += nextBlock
// 		}

// 		for j := 0; j < tau; j++ {
// 			dfsId[j] = tmp[j]
// 		}
// 	}

// 	galEls := make([]uint64, int(math.Log2(float64(tau))))
// 	for i := 0; i < int(math.Log2(float64(tau))); i++ {
// 		galEls[i] = uint64(tau/int(math.Pow(2, float64(i))) + 1)
// 	}
// 	rotkey := kgen.GenRotationKeys(galEls, sk)

// 	evkRGSW := &rlwe.EvaluationKey{Rlk: rlk}
// 	evkRLWE := &rlwe.EvaluationKey{Rlk: rlk, Rtks: rotkey}

// 	encryptorRLWE := rlwe.NewEncryptor(params, sk)
// 	decryptorRLWE := rlwe.NewDecryptor(params, sk)
// 	encryptorRGSW := rgsw.NewEncryptor(params, sk)
// 	evaluatorRGSW := rgsw.NewEvaluator(params, evkRGSW)
// 	evaluatorRLWE := rlwe.NewEvaluator(params, evkRLWE)

// 	levelQ := params.QCount() - 1
// 	levelP := params.PCount() - 1
// 	decompRNS := params.DecompRNS(levelQ, levelP)
// 	decompPw2 := params.DecompPw2(levelQ, levelP)
// 	ringQP := params.RingQP()
// 	ringQ := params.RingQ()

// 	// ======== Set Scale factors ========
// 	s := 1 / 1000.0
// 	L := 1 / 100000.0
// 	r := 1 / 1000.0

// 	// ======== Number of iterations ========
// 	iter := 50

// 	// ======== Plant matrices ========
// 	A := [][]float64{
// 		{1, 0.0020, 0.0663, 0.0047, 0.0076},
// 		{0, 1.0077, 2.0328, -0.5496, -0.0591},
// 		{0, 0.0478, 0.9850, -0.0205, -0.0092},
// 		{0, 0, 0, 0.3679, 0},
// 		{0, 0, 0, 0, 0.3679},
// 	}
// 	B := [][]float64{
// 		{0.0029, 0.0045},
// 		{-0.3178, -0.0323},
// 		{-0.0086, -0.0051},
// 		{0.6321, 0},
// 		{0, 0.6321},
// 	}
// 	C := [][]float64{
// 		{0, 1, 0, 0, 0},
// 		{0, -0.2680, 47.7600, -4.5600, 4.4500},
// 		{1, 0, 0, 0, 0},
// 		{0, 0, 0, 1, 0},
// 		{0, 0, 0, 0, 1},
// 	}

// 	// ======== Controller matrices ========
// 	// F: n x n
// 	// G: n x p
// 	// H: m x n
// 	// R: n x m

// 	F := [][]float64{ // Must be an integer matrix
// 		{2, 0, 0, 0, 0},
// 		{0, -1, 0, 0, 0},
// 		{0, 0, 1, 0, 0},
// 		{0, 0, 0, 0, 0},
// 		{0, 0, 0, 0, 0},
// 	}

// 	G := [][]float64{
// 		{0.0816, 0.0047, 1.6504, -0.0931, 0.4047},
// 		{-1.4165, -0.3163, -0.4329, 0.1405, 0.8263},
// 		{-1.4979, -0.2089, -0.6394, 0.3682, 0.7396},
// 		{0.0459, 0.0152, 1.1004, -0.1187, 0.6563},
// 		{0.0020, 0.0931, 0.0302, -0.0035, 0.0177},
// 	}

// 	R := [][]float64{
// 		{-3.5321, 23.1563},
// 		{-0.5080, -2.3350},
// 		{2.5496, 0.9680},
// 		{0.0436, -1.1227},
// 		{-0.7560, 0.7144},
// 	}

// 	H := [][]float64{
// 		{0.0399, -0.3269, -0.2171, 0.0165, -0.0655},
// 		{-0.0615, -0.0492, -0.0322, -0.0077, -0.0084},
// 	}

// 	// ======== Scale up G, R, and H to integers ========
// 	Gbar := scalarMatMult(1/s, G)
// 	Rbar := scalarMatMult(1/s, R)
// 	Hbar := scalarMatMult(1/s, H)

// 	// ======== Plant and Controller initial state ========
// 	xPlantInit := []float64{
// 		1,
// 		-1,
// 		0,
// 		0.7,
// 		2,
// 	}
// 	xContInit := []float64{
// 		-0.001,
// 		0.013,
// 		0.2,
// 		-0.02,
// 		0,
// 	}

// 	// ======== F,G,H,R RGSW encrpytion ========
// 	n := len(F)
// 	p_ := len(G[0])
// 	m := len(H)
// 	fmt.Printf("n \n %d\n", n)
// 	fmt.Printf("p_ \n %d \n", p_)
// 	fmt.Printf("m \n %d \n", m)

// 	// Dimension: 1-by-(# of columns)
// 	ctF := encryptRgsw(F, tau, encryptorRGSW, levelQ, levelP, decompRNS, decompPw2, ringQ, ringQP, params)
// 	ctG := encryptRgsw(Gbar, tau, encryptorRGSW, levelQ, levelP, decompRNS, decompPw2, ringQ, ringQP, params)
// 	ctH := encryptRgsw(Hbar, tau, encryptorRGSW, levelQ, levelP, decompRNS, decompPw2, ringQ, ringQP, params)
// 	ctR := encryptRgsw(Rbar, tau, encryptorRGSW, levelQ, levelP, decompRNS, decompPw2, ringQ, ringQP, params)

// 	// ======== Run closed-loop without encryption ========
// 	fmt.Println("Nominal Loop Start")

// 	// State and output storage
// 	YOUT := [][]float64{}
// 	UOUT := [][]float64{}
// 	XCONT := [][]float64{}
// 	XPLANT := [][]float64{}

// 	// State initialization
// 	xPlantUnenc := xPlantInit
// 	xContUnenc := xContInit

// 	startUnenc := time.Now()
// 	for i := 0; i < iter; i++ {
// 		// Plant output
// 		yOut := matVecMult(C, xPlantUnenc)

// 		// Controller output
// 		uOut := matVecMult(H, xContUnenc)

// 		// Plant state update
// 		xPlantUnenc = vecAdd(matVecMult(A, xPlantUnenc), matVecMult(B, uOut))

// 		// Controller state update
// 		xContUnenc = vecAdd(matVecMult(F, xContUnenc), matVecMult(G, yOut))
// 		xContUnenc = vecAdd(xContUnenc, matVecMult(R, uOut))

// 		// Append data
// 		YOUT = append(YOUT, yOut)
// 		UOUT = append(UOUT, uOut)
// 		XCONT = append(XCONT, xContUnenc)
// 		XPLANT = append(XPLANT, xPlantUnenc)

// 	}
// 	elapsedUnenc := time.Now().Sub(startUnenc)

// 	// ======== Run closed-loop with encryption ========
// 	fmt.Println("Encrypted Loop Start")

// 	// State and output storage
// 	YOUTENC := [][]float64{}
// 	UOUTENC := [][]float64{}
// 	XCONTENC := [][]float64{}
// 	XPLANTENC := [][]float64{}

// 	// State initialization
// 	// Dimension: 1-by-(# of elements)
// 	xPlantEnc := xPlantInit
// 	xContEnc := scalarVecMult(1/(r*s), xContInit)
// 	ctxCont := encryptRlwe(xContEnc, 1/L, encryptorRLWE, ringQ, params)
// 	ctTmp := rlwe.NewCiphertext(params, ctxCont[0].Degree(), ctxCont[0].Level())

// 	startEnc := time.Now()
// 	for i := 0; i < iter; i++ {
// 		// Plant output
// 		yOut := matVecMult(C, xPlantEnc)

// 		// Quantize and encrypt plant output
// 		yOutRound := roundVec(scalarVecMult(1/r, yOut))
// 		ctyOut := encryptRlwe(yOutRound, 1/L, encryptorRLWE, ringQ, params)

// 		// Decrypt plant output just for validation
// 		valyOut := decryptNewRlwe(ctyOut, decryptorRLWE, r*L, ringQ, params)

// 		// Controller output
// 		packedCtuOut := externalProduct(ctxCont, ctH, evaluatorRGSW, ringQ, params)

// 		// Unpack controller output
// 		unpackedCtuOut := unpackPackedCt(packedCtuOut, m, tau, evaluatorRLWE, ringQ, params)

// 		// Decrypt controller output and construct plant input
// 		uOut := decryptNewRlwe(unpackedCtuOut, decryptorRLWE, r*s*s*L, ringQ, params)

// 		// Re-encrypt controller output
// 		ctuReEnc := encryptRlwe(uOut, 1/(r*L), encryptorRLWE, ringQ, params)

// 		// Plant state update
// 		xPlantEnc = vecAdd(matVecMult(A, xPlantEnc), matVecMult(B, uOut))

// 		// Controller state update
// 		ctFx := externalProduct(ctxCont, ctF, evaluatorRGSW, ringQ, params)
// 		ctGy := externalProduct(ctyOut, ctG, evaluatorRGSW, ringQ, params)
// 		ctRu := externalProduct(ctuReEnc, ctR, evaluatorRGSW, ringQ, params)
// 		ctTmp = ctAdd(ctFx, ctGy, params)
// 		ctTmp = ctAdd(ctTmp, ctRu, params)
// 		ctxCont = unpackPackedCt(ctTmp, n, tau, evaluatorRLWE, ringQ, params)

// 		// Decrypt controller state just for validation
// 		valxCont := decryptNewRlwe(ctxCont, decryptorRLWE, r*s*L, ringQ, params)

// 		// Append data
// 		YOUTENC = append(YOUTENC, valyOut)
// 		UOUTENC = append(UOUTENC, uOut)
// 		XCONTENC = append(XCONTENC, valxCont)
// 		XPLANTENC = append(XPLANTENC, xPlantEnc)
// 	}
// 	elapsedEnc := time.Now().Sub(startEnc)

// 	// ======== For debugging ========
// 	// for i := 0; i < iter; i++ {
// 	// 	fmt.Println("=======================")
// 	// 	fmt.Printf("Plant output at iter %d: \n %v \n", i, YOUT[i])
// 	// 	fmt.Printf("Controller output at iter %d: \n %v \n", i, UOUT[i])
// 	// 	fmt.Printf("Plant state at iter %d: \n %v \n", i, XPLANT[i])
// 	// 	fmt.Printf("Controller state at iter %d: \n %v \n", i, XCONT[i])
// 	// 	fmt.Println("=======================")
// 	// 	fmt.Printf("Decrypted Plant output at iter %d: \n %v \n", i, YOUTENC[i])
// 	// 	fmt.Printf("Decrypted controller output at iter %d: \n %v \n", i, UOUTENC[i])
// 	// 	fmt.Printf("Decrypted Plant state at iter %d: \n %v \n", i, XPLANTENC[i])
// 	// 	fmt.Printf("Decrypted controller state at iter %d: \n %v \n", i, XCONTENC[i])
// 	// }

// 	// ======== Simulation result ========
// 	fmt.Println("Iterations: ", iter)
// 	fmt.Println("Unenc total time: ", elapsedUnenc)
// 	fmt.Println("Enc total time: ", elapsedEnc)
// 	fmt.Printf("Unenc average time for one iteration: %v us \n", float64(elapsedUnenc.Microseconds())/float64(iter))
// 	fmt.Printf("Enc average time for one iteration: %v us \n", float64(elapsedEnc.Microseconds())/float64(iter))

// 	// **************************** Plot section *************************************
// 	// Plot 2-norm of the difference between plant state, controller state, plant output, and controller output
// 	rows := 2
// 	cols := 2
// 	plotsDiff := make([][]*plot.Plot, rows)
// 	for i := 0; i < rows; i++ {
// 		plotsDiff[i] = make([]*plot.Plot, cols)
// 		for j := 0; j < cols; j++ {
// 			p := plot.New()

// 			pts := make(plotter.XYs, iter)

// 			for k := range pts {
// 				pts[k].X = float64(k)
// 			}

// 			// First row
// 			if i == 0 {
// 				if j == 0 {
// 					for k := range pts {
// 						pts[k].Y = vec2norm(vecSubtract(XPLANT[k], XPLANTENC[k]))

// 					}
// 					p.Title.Text = "Plant State Difference"
// 				} else if j == 1 {
// 					for k := range pts {
// 						pts[k].Y = vec2norm(vecSubtract(YOUT[k], YOUTENC[k]))
// 					}
// 					p.Title.Text = "Plant Output Difference"
// 				}
// 			} else {

// 				// Second row
// 				if j == 0 {
// 					for k := range pts {
// 						pts[k].Y = vec2norm(vecSubtract(XCONT[k], XCONTENC[k]))

// 					}
// 					p.Title.Text = "Controller State Difference"
// 				} else if j == 1 {
// 					for k := range pts {
// 						pts[k].Y = vec2norm(vecSubtract(UOUT[k], UOUTENC[k]))
// 					}
// 					p.Title.Text = "Controller Output Difference"
// 				}
// 			}
// 			p.X.Label.Text = "iteration"
// 			p.Y.Label.Text = "2-norm value"
// 			// p.Y.Min = -1e-04
// 			// p.Y.Max = 1e-04
// 			p.Add(plotter.NewGrid())
// 			lLine, lPoints, _ := plotter.NewLinePoints(pts)
// 			p.Add(lLine, lPoints)
// 			plotsDiff[i][j] = p

// 		}
// 	}

// 	imgDiff := vgimg.New(vg.Points(1000), vg.Points(500))
// 	dcDiff := draw.New(imgDiff)

// 	t := draw.Tiles{
// 		Rows:      rows,
// 		Cols:      cols,
// 		PadX:      vg.Millimeter,
// 		PadY:      vg.Millimeter,
// 		PadTop:    vg.Points(50),
// 		PadBottom: vg.Points(50),
// 		PadLeft:   vg.Points(50),
// 		PadRight:  vg.Points(50),
// 	}

// 	canvasesDiff := plot.Align(plotsDiff, t, dcDiff)
// 	for i := 0; i < rows; i++ {
// 		for j := 0; j < cols; j++ {
// 			if plotsDiff[i][j] != nil {
// 				plotsDiff[i][j].Draw(canvasesDiff[i][j])
// 			}
// 		}
// 	}

// 	w, err := os.Create("Difference.png")
// 	if err != nil {
// 		panic(err)
// 	}
// 	defer w.Close()
// 	png := vgimg.PngCanvas{Canvas: imgDiff}
// 	if _, err := png.WriteTo(w); err != nil {
// 		panic(err)
// 	}
// }