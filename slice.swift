

import TensorFlow

enum Slice : Equatable {
    case slice(Int, Int)
    case from(Int)
    case to(Int)
    case all
    case rest

    func get_range(_ n: Int) -> (Int, Int) {
        var low: Int = 0
        var high: Int = n

        switch self {
            case .slice(let _low, let _high):
                low = _low
                high = _high
            case .from(let _low):
                low = _low
            case .to(let _high):
                high = _high
            default:
                break 
        }


        if low < 0 {
            low += n
        }

        if high < 0 {
            high += n
        }

        return (low, high)
    }
}

enum SliceError : Error {
    case MultipleRest
    case TooManySlices
}

extension Tensor {

    func slice(_ slices: Slice...) throws -> Tensor {
        var slices = slices
        let ndim = Int(self.shape.count)

        if slices.count > ndim {
            throw SliceError.TooManySlices
        }

        if slices.count < ndim && !slices.contains(.rest) {
            slices.append(.rest)
        }

        let restCount = slices.filter { $0 == .rest }.count
        let notRestCount = slices.count - restCount
        let restExpand = ndim - notRestCount

        if restCount > 1 {
            throw SliceError.MultipleRest
        }

        if restExpand > 0 {
            let allSlices = Array(repeating: Slice.all, count: restExpand)
            let idx = slices.firstIndex(of: .rest)!

            slices = slices[..<idx] + allSlices + slices[(idx+1)...]
        }

        var lowerBounds: [Int32] = []
        var upperBounds: [Int32] = []

        for (slice, n) in zip(slices, self.shape.dimensions) {
            let (lower, higher) = slice.get_range(Int(n))
            lowerBounds.append(Int32(lower))
            upperBounds.append(Int32(higher))
        }

        print(lowerBounds)
        print(upperBounds)

        return self.slice(lowerBounds: lowerBounds, upperBounds: upperBounds)
    }
}


var x = Tensor<Float>(repeating: 0, shape: [10, 6, 4, 5, 8])

x = try x.slice(.to(4), .all, .slice(3, 5), .all, .all)

print(x.shape)


